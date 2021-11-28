#include "aesni_prf_processors.h"
#include "aesni_helpers.h"

#include <cassert>

#include <wmmintrin.h>
#include <tmmintrin.h>
#include <smmintrin.h>
#include <emmintrin.h>

#include <iostream>
#include <iomanip>

constexpr size_t mainEvaluatingWidthXor = 4;
constexpr size_t mainEvaluatingWidthAnd = 4; // 4
constexpr size_t mainGarblingWidthAnd = 2;
constexpr size_t mainGarblingWidthXor = 3;

void PRFXorLTEvaluatingAesniProcessor::computeAESOutKeys(uint32_t tableCounter, size_t numTablesInBatch, uint8_t* receivedTables)
{
	ProcessQueue(m_gateQueue, numTablesInBatch, tableCounter, receivedTables);
}

size_t PRFXorLTEvaluatingAesniProcessor::vectorWidth() const
{
	return mainEvaluatingWidthXor;
}

void PRFXorLTEvaluatingAesniProcessor::BulkProcessor(uint32_t wireCounter, size_t numWiresInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer)
{
	computeAESOutKeys<mainEvaluatingWidthXor>(wireCounter, queueStartIndex, simdStartOffset, numWiresInBatch, tableBuffer);
}

void PRFXorLTEvaluatingAesniProcessor::LeftoversProcessor(uint32_t wireCounter, size_t numWiresInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer)
{
	computeAESOutKeys<1>(wireCounter, queueStartIndex, simdStartOffset, numWiresInBatch, tableBuffer);
}

template<size_t width>
void PRFXorLTEvaluatingAesniProcessor::computeAESOutKeys(uint32_t tableCounter, size_t queueStartIndex, size_t simdStartOffset, size_t numTablesInBatch, const uint8_t* receivedTables)
{
	constexpr size_t num_keys = 2 * width;

	// note: this implementation heavily relies on the fact that
	// the optimizer notices that all w-indexed loop iterations are independent
	// *and* that it manages to assign different registers to each iteration

	__m128i data[num_keys];
	__m128i parentKeys[num_keys];
	__m128i finalMask[width];
	uint8_t* targetGateKey[width];
	uint8_t opbit[width];
	bool doRXor[width];
	const uint8_t* gtptr = receivedTables + tableCounter * 16;

	// we need to encode the signal bits as part of the key
	// so we need to strip them before AES processing
	// which this constant allows
	const __m128i signalBitCleaner = _mm_set_epi8(0xFE, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);


	size_t currentGateIdx = queueStartIndex;
	uint32_t currentOffset = simdStartOffset;

	for (size_t i = 0; i < numTablesInBatch; i += width)
	{
		for (size_t w = 0; w < width; ++w)
		{
			const GATE* currentGate = m_gateQueue[currentGateIdx];
			const uint32_t leftParentId = currentGate->ingates.inputs.twin.left;
			const uint32_t rightParentId = currentGate->ingates.inputs.twin.right;
			const GATE* leftParent = &m_vGates[leftParentId];
			const GATE* rightParent = &m_vGates[rightParentId];
			const uint8_t* leftParentKey = leftParent->gs.yval + 16 * currentOffset;
			const uint8_t* rightParentKey = rightParent->gs.yval + 16 * currentOffset;

			parentKeys[2 * w + 0] = _mm_loadu_si128((__m128i*)leftParentKey);
			parentKeys[2 * w + 0] = _mm_and_si128(signalBitCleaner, parentKeys[2 * w + 0]);
			parentKeys[2 * w + 1] = _mm_loadu_si128((__m128i*)rightParentKey);
			parentKeys[2 * w + 1] = _mm_and_si128(signalBitCleaner, parentKeys[2 * w + 1]);

			const uint8_t lpbit = leftParentKey[15] & 0x01;
			const uint8_t rpbit = rightParentKey[15] & 0x01; 
			opbit[w] = lpbit ^ rpbit;

			data[2 * w + 0] = _mm_set_epi64x(lpbit, m_vWireIds[currentGateIdx] + currentOffset);
			data[2 * w + 1] = _mm_set_epi64x(rpbit, m_vWireIds[currentGateIdx] + currentOffset);

			/*
			std::cout << std::endl << std::endl;

			
			std::cout << "lParent: ";
			PrintKey(parentKeys[2 * w + 0]);
			std::cout << std::endl;
			std::cout << "rParent: ";
			PrintKey(parentKeys[2 * w + 1]);
			std::cout << std::endl;
			*/
			

			targetGateKey[w] = currentGate->gs.yval + 16 * currentOffset;

			finalMask[w] = _mm_setzero_si128();
			if (rpbit)
			{
				finalMask[w] = _mm_loadu_si128((__m128i*)gtptr);
				doRXor[w] = true;
			}
			else
			{
				finalMask[w] = parentKeys[2 * w + 1];
				doRXor[w] = false;
			}
			gtptr += 16;

			//std::cout << "Bits: " << (uint16_t)lpbit << (uint16_t)rpbit << std::endl;

			currentOffset++;

			if (currentOffset >= currentGate->nvals)
			{
				currentGateIdx++;
				currentOffset = 0;
			}
		}

		aesni_encrypt_variable_keys<num_keys, 1>(parentKeys, data);

		for (size_t w = 0; w < width; ++w)
		{
			
			/*
			std::cout << "Pre-Mask:   ";
			PrintKey(data[2 * w + 0]);
			std::cout << std::endl;
			*/
			
			__m128i temp = _mm_xor_si128(data[2 * w + 0], finalMask[w]);
			
			/*
			std::cout << "Post-Mask:  ";
			PrintKey(temp);
			std::cout << std::endl;
			*/
			
			if(doRXor[w])
				temp = _mm_xor_si128(temp, data[2 * w + 1]);
			
			/*
			std::cout << "Post-XOR:   ";
			PrintKey(temp);
			std::cout << std::endl;
			*/
			
			_mm_storeu_si128((__m128i*)(targetGateKey[w]), temp);
			targetGateKey[w][15] = (targetGateKey[w][15] & 0xFE) | opbit[w]; // clear out the permutation bit and then set it
			
			/*
			std::cout << "Stored Key: ";
			PrintKey(_mm_loadu_si128((__m128i*)targetGateKey[w]));
			std::cout << std::endl;
			*/
		}
	}
}

void PRFXorLTGarblingAesniProcessor::computeAESOutKeys(uint32_t tableCounter, size_t numTablesInBatch, uint8_t* receivedTables)
{
	ProcessQueue(m_gateQueue, numTablesInBatch, tableCounter, receivedTables);
}

size_t PRFXorLTGarblingAesniProcessor::vectorWidth() const
{
	return mainGarblingWidthXor;
}

void PRFXorLTGarblingAesniProcessor::BulkProcessor(uint32_t wireCounter, size_t numWiresInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer)
{
	computeAESOutKeys<mainGarblingWidthXor>(wireCounter, queueStartIndex, simdStartOffset, numWiresInBatch, tableBuffer);
}

void PRFXorLTGarblingAesniProcessor::LeftoversProcessor(uint32_t wireCounter, size_t numWiresInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer)
{
	computeAESOutKeys<1>(wireCounter, queueStartIndex, simdStartOffset, numWiresInBatch, tableBuffer);
}

template<size_t width>
void PRFXorLTGarblingAesniProcessor::computeAESOutKeys(uint32_t tableCounter, size_t queueStartIndex, size_t simdStartOffset, size_t numTablesInBatch, uint8_t* receivedTables)
{
	constexpr size_t num_keys = 3 * width;

	// note: this implementation heavily relies on the fact that
	// the optimizer notices that all w-indexed loop iterations are independent
	// *and* that it manages to assign different registers to each iteration

	__m128i data[num_keys];
	__m128i parentKeys[num_keys];
	__m128i rkey[width];
	uint8_t* targetGateKey0[width];
	uint8_t* targetGateKey1[width];
	uint8_t rpbit[width];
	uint8_t* gtptr = receivedTables + tableCounter * 16;
	const __m128i signalBitCleaner = _mm_set_epi8(0xFE, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);

	size_t currentGateIdx = queueStartIndex;
	uint32_t currentOffset = simdStartOffset;

	for (size_t i = 0; i < numTablesInBatch; i += width)
	{
		for (size_t w = 0; w < width; ++w)
		{
			const GATE* currentGate = m_gateQueue[currentGateIdx];
			const uint32_t leftParentId = currentGate->ingates.inputs.twin.left;
			const uint32_t rightParentId = currentGate->ingates.inputs.twin.right;
			const GATE* leftParent = &m_vGates[leftParentId];
			const GATE* rightParent = &m_vGates[rightParentId];
			const uint8_t* leftParentKey0 = leftParent->gs.yinput.outKey[0] + 16 * currentOffset;
			const uint8_t* leftParentKey1 = leftParent->gs.yinput.outKey[1] + 16 * currentOffset;
			const uint8_t* rightParentKey0 = rightParent->gs.yinput.outKey[0] + 16 * currentOffset;
			const uint8_t* rightParentKey1 = rightParent->gs.yinput.outKey[1] + 16 * currentOffset;
			const uint8_t lpbit = leftParent->gs.yinput.pi[currentOffset];
			rpbit[w] = rightParent->gs.yinput.pi[currentOffset];
			currentGate->gs.yinput.pi[currentOffset] = lpbit ^ rpbit[w];

			parentKeys[3 * w + 0] = _mm_loadu_si128((__m128i*)leftParentKey0);
			parentKeys[3 * w + 1] = _mm_loadu_si128((__m128i*)leftParentKey1);

			if (rpbit[w]) {
				parentKeys[3 * w + 2] = _mm_loadu_si128((__m128i*)rightParentKey0);
				rkey[w] = _mm_loadu_si128((__m128i*)rightParentKey1);
			}	
			else {
				rkey[w] = _mm_loadu_si128((__m128i*)rightParentKey0);
				parentKeys[3 * w + 2] = _mm_loadu_si128((__m128i*)rightParentKey1);
			}

			parentKeys[3 * w + 0] = _mm_and_si128(signalBitCleaner, parentKeys[3 * w + 0]);
			parentKeys[3 * w + 1] = _mm_and_si128(signalBitCleaner, parentKeys[3 * w + 1]);
			parentKeys[3 * w + 2] = _mm_and_si128(signalBitCleaner, parentKeys[3 * w + 2]);
			rkey[w] = _mm_and_si128(signalBitCleaner, rkey[w]);
				
			/*
			std::cout << std::endl << std::endl;

			
			std::cout << "Bits: " << (uint16_t)lpbit << (uint16_t)rpbit[w] << std::endl;

			std::cout << "lParent0: ";
			PrintKey(parentKeys[3 * w + 0]);
			std::cout << std::endl;
			std::cout << "lParent1: ";
			PrintKey(parentKeys[3 * w + 1]);
			std::cout << std::endl;
			std::cout << "rParentP: ";
			PrintKey(parentKeys[3 * w + 2]);
			std::cout << std::endl;
			std::cout << "rParentD: ";
			PrintKey(rkey[w]);
			std::cout << std::endl;
			*/
			

			data[3 * w + 0] = _mm_set_epi64x(lpbit, m_vWireIds[currentGateIdx] + currentOffset);
			data[3 * w + 1] = _mm_set_epi64x(1-lpbit, m_vWireIds[currentGateIdx] + currentOffset);
			data[3 * w + 2] = _mm_set_epi64x(1, m_vWireIds[currentGateIdx] + currentOffset);

			targetGateKey0[w] = currentGate->gs.yinput.outKey[0] + 16 * currentOffset;
			targetGateKey1[w] = currentGate->gs.yinput.outKey[1] + 16 * currentOffset;

			currentOffset++;

			if (currentOffset >= currentGate->nvals)
			{
				currentGateIdx++;
				currentOffset = 0;
			}
		}

		aesni_encrypt_variable_keys<num_keys, 1>(parentKeys, data);

		for (size_t w = 0; w < width; ++w)
		{
			/*
			std::cout << "Left Delta Cipher:  ";
			PrintKey(data[3 * w + 0]);
			std::cout << std::endl;
			std::cout << "Right Delta Cipher: ";
			PrintKey(data[3 * w + 1]);
			std::cout << std::endl;
			*/

			__m128i delta = _mm_xor_si128(data[3 * w + 0], data[3 * w + 1]);

			__m128i tableMask = _mm_xor_si128(rkey[w], delta);
			__m128i tableEntry = _mm_xor_si128(data[3 * w + 2], tableMask);

			__m128i rtkey0;
			if (rpbit[w]) {
				rtkey0 = tableMask;
			}
			else {
				rtkey0 = rkey[w];
			}
			_mm_storeu_si128((__m128i*)gtptr, tableEntry);
			gtptr += 16;
			/*
			std::cout << "Right Trans Key:    ";
			PrintKey(rtkey0);
			std::cout << std::endl;
			*/
			__m128i outkey0 = _mm_xor_si128(data[3 * w + 0], rtkey0);
			__m128i outkey1 = _mm_xor_si128(data[3 * w + 1], rtkey0);
			_mm_storeu_si128((__m128i*)targetGateKey0[w], outkey0);
			_mm_storeu_si128((__m128i*)targetGateKey1[w], outkey1);
			/*
			std::cout << "Stored Keys:" << std::endl;
			PrintKey(_mm_loadu_si128((__m128i*)targetGateKey0[w]));
			std::cout << std::endl;
			PrintKey(_mm_loadu_si128((__m128i*)targetGateKey1[w]));
			std::cout << std::endl;
			*/
		}
	}
}

void PRFAndLTEvaluatingAesniProcessor::computeAESOutKeys(uint32_t tableCounter, size_t numTablesInBatch, uint8_t* receivedTables)
{
	ProcessQueue(m_gateQueue, numTablesInBatch, tableCounter, receivedTables);
}

size_t PRFAndLTEvaluatingAesniProcessor::vectorWidth() const
{
	return mainEvaluatingWidthAnd;
}

void PRFAndLTEvaluatingAesniProcessor::BulkProcessor(uint32_t wireCounter, size_t numWiresInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer)
{
	computeAESOutKeys<mainEvaluatingWidthAnd>(wireCounter, queueStartIndex, simdStartOffset, numWiresInBatch, tableBuffer);
}

void PRFAndLTEvaluatingAesniProcessor::LeftoversProcessor(uint32_t wireCounter, size_t numWiresInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer)
{
	computeAESOutKeys<1>(wireCounter, queueStartIndex, simdStartOffset, numWiresInBatch, tableBuffer);
}

template<size_t width>
void PRFAndLTEvaluatingAesniProcessor::computeAESOutKeys(uint32_t tableCounter, size_t queueStartIndex, size_t simdStartOffset, size_t numTablesInBatch, const uint8_t* receivedTables)
{
	constexpr size_t numkeys = 2 * width;

	// note: this implementation heavily relies on the fact that
	// the optimizer notices that all w-indexed loop iterations are independent
	// *and* that it manages to assign different registers to each iteration

	__m128i data[numkeys];
	__m128i parentKeys[numkeys];
	__m128i finalMask[width];
	uint8_t* targetGateKey[width];
	uint8_t opbit[width];
	const uint8_t* gtptr = receivedTables + tableCounter * 2 * 16;
	const __m128i signalBitCleaner = _mm_set_epi8(0xFE, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);


	size_t currentGateIdx = queueStartIndex;
	uint32_t currentOffset = simdStartOffset;

	for (size_t i = 0; i < numTablesInBatch; i += width)
	{
		for (size_t w = 0; w < width; ++w)
		{
			const GATE* currentGate = m_gateQueue[currentGateIdx];
			const uint32_t leftParentId = currentGate->ingates.inputs.twin.left;
			const uint32_t rightParentId = currentGate->ingates.inputs.twin.right;
			const GATE* leftParent = &m_vGates[leftParentId];
			const GATE* rightParent = &m_vGates[rightParentId];
			const uint8_t* leftParentKey = leftParent->gs.yval + 16 * currentOffset;
			const uint8_t* rightParentKey = rightParent->gs.yval + 16 * currentOffset;

			parentKeys[2 * w + 0] = _mm_loadu_si128((__m128i*)leftParentKey);
			parentKeys[2 * w + 1] = _mm_loadu_si128((__m128i*)rightParentKey);

			parentKeys[2 * w + 0] = _mm_and_si128(signalBitCleaner, parentKeys[2 * w + 0]);
			parentKeys[2 * w + 1] = _mm_and_si128(signalBitCleaner, parentKeys[2 * w + 1]);

			/*
			std::cout << std::endl << std::endl;

			
			std::cout << "ParentKeys: " << std::endl;
			PrintKey(parentKeys[2 * w + 0]);
			std::cout << std::endl;
			PrintKey(parentKeys[2 * w + 1]);
			std::cout << std::endl;
			*/


			const uint8_t lpbit = leftParentKey[15] & 0x01;
			const uint8_t rpbit = rightParentKey[15] & 0x01;
			const uint8_t combined_bits = (lpbit << 1) | rpbit;

			//std::cout << "parent bits: " << (uint16_t)lpbit << (uint16_t)rpbit << std::endl;

			data[2 * w + 0] = _mm_set_epi64x(combined_bits, m_vWireIds[currentGateIdx] + currentOffset);
			data[2 * w + 1] = _mm_set_epi64x(combined_bits, m_vWireIds[currentGateIdx] + currentOffset);

			targetGateKey[w] = currentGate->gs.yval + 16 * currentOffset;

			__m128i fcmask = _mm_set_epi8(0xFC, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
			finalMask[w] = _mm_setzero_si128();	
			const uint8_t transmittedBitFirst = gtptr[15] & 0x03;
			__m128i firstTable = _mm_loadu_si128((__m128i*)gtptr);
			firstTable = _mm_and_si128(fcmask, firstTable);
			if (rpbit & 1)
			{
				finalMask[w] = firstTable;
			}
			gtptr += 16;
			const uint8_t transmittedBitSecond = gtptr[15] & 0x03;
			__m128i secondTable = _mm_loadu_si128((__m128i*)gtptr);
			secondTable = _mm_and_si128(fcmask, secondTable);
			if (lpbit & 1)
			{
				finalMask[w] = _mm_xor_si128(secondTable, finalMask[w]);
			}
			gtptr += 16;

			const uint8_t joinedBits = (transmittedBitSecond << 2) | transmittedBitFirst;
			opbit[w] = (joinedBits >> combined_bits) & 0x01;

			//std::cout << "Combined bits: " << (uint16_t) combined_bits << std::endl;

			currentOffset++;

			if (currentOffset >= currentGate->nvals)
			{
				currentGateIdx++;
				currentOffset = 0;
			}
		}

		aesni_encrypt_variable_keys<numkeys, 1>(parentKeys, data);

		for (size_t w = 0; w < width; ++w)
		{
			__m128i temp = _mm_xor_si128(data[2 * w + 0], data[2 * w + 1]);
			//std::cout << "XOR Key:" << std::endl;
			//PrintKey(temp);
			//std::cout << std::endl;
			__m128i mask = _mm_set_epi8(0xFD, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
			temp = _mm_and_si128(mask, temp);
			temp = _mm_xor_si128(temp, finalMask[w]);
			_mm_storeu_si128((__m128i*)(targetGateKey[w]), temp);
			targetGateKey[w][15] ^= opbit[w]; // clear out the permutation bit and then set it
			//std::cout << "Stored Key:" << std::endl;
			//PrintKey(_mm_loadu_si128((__m128i*)targetGateKey[w]));
			//std::cout << std::endl;
		}
	}
}

void PRFAndLTGarblingAesniProcessor::computeAESOutKeys(uint32_t tableCounter, size_t numTablesInBatch, uint8_t* receivedTables)
{
	ProcessQueue(m_gateQueue, numTablesInBatch, tableCounter, receivedTables);
}

size_t PRFAndLTGarblingAesniProcessor::vectorWidth() const
{
	return mainGarblingWidthAnd;
}

void PRFAndLTGarblingAesniProcessor::BulkProcessor(uint32_t wireCounter, size_t numWiresInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer)
{
	computeAESOutKeys<mainGarblingWidthAnd>(wireCounter, queueStartIndex, simdStartOffset, numWiresInBatch, tableBuffer);
}

void PRFAndLTGarblingAesniProcessor::LeftoversProcessor(uint32_t wireCounter, size_t numWiresInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer)
{
	computeAESOutKeys<1>(wireCounter, queueStartIndex, simdStartOffset, numWiresInBatch, tableBuffer);
}


template<size_t width>
void PRFAndLTGarblingAesniProcessor::computeAESOutKeys(uint32_t tableCounter, size_t queueStartIndex, size_t simdStartOffset, size_t numTablesInBatch, uint8_t* receivedTables)
{
	constexpr size_t num_keys = 4 * width;

	// note: this implementation heavily relies on the fact that
	// the optimizer notices that all w-indexed loop iterations are independent
	// *and* that it manages to assign different registers to each iteration

	__m128i data[2* num_keys];
	__m128i parentKeys[num_keys]; // lpi, l!pi, rpi, r!pi
	uint8_t* targetGateKey0[width];
	uint8_t* targetGateKey1[width];
	uint8_t combined_bits[width];
	uint8_t opbit[width];
	uint8_t* gtptr = receivedTables + tableCounter * 2 * 16;
	const __m128i signalBitCleaner = _mm_set_epi8(0xFE, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);

	size_t currentGateIdx = queueStartIndex;
	uint32_t currentOffset = simdStartOffset;

	for (size_t i = 0; i < numTablesInBatch; i += width)
	{
		for (size_t w = 0; w < width; ++w)
		{
			const GATE* currentGate = m_gateQueue[currentGateIdx];
			const uint32_t leftParentId = currentGate->ingates.inputs.twin.left;
			const uint32_t rightParentId = currentGate->ingates.inputs.twin.right;
			const GATE* leftParent = &m_vGates[leftParentId];
			const GATE* rightParent = &m_vGates[rightParentId];
			const uint8_t* leftParentKey0 = leftParent->gs.yinput.outKey[0] + 16 * currentOffset;
			const uint8_t* leftParentKey1 = leftParent->gs.yinput.outKey[1] + 16 * currentOffset;
			const uint8_t* rightParentKey0 = rightParent->gs.yinput.outKey[0] + 16 * currentOffset;
			const uint8_t* rightParentKey1 = rightParent->gs.yinput.outKey[1] + 16 * currentOffset;
			const uint8_t lpbit = leftParent->gs.yinput.pi[currentOffset];
			const uint8_t rpbit = rightParent->gs.yinput.pi[currentOffset];
			assert(lpbit < 2);
			assert(rpbit < 2);
			opbit[w] = currentGate->gs.yinput.pi[currentOffset];
			combined_bits[w] = ((1 - lpbit) << 1) | (1 - rpbit);

			
			if (lpbit) {
				parentKeys[4 * w + 0] = _mm_loadu_si128((__m128i*)leftParentKey1);
				parentKeys[4 * w + 1] = _mm_loadu_si128((__m128i*)leftParentKey0);
			}
			else {
				parentKeys[4 * w + 1] = _mm_loadu_si128((__m128i*)leftParentKey1);
				parentKeys[4 * w + 0] = _mm_loadu_si128((__m128i*)leftParentKey0);	
			}
			
			
			
			if (rpbit) {
				parentKeys[4 * w + 2] = _mm_loadu_si128((__m128i*)rightParentKey1);
				parentKeys[4 * w + 3] = _mm_loadu_si128((__m128i*)rightParentKey0);
			}
			else {
				parentKeys[4 * w + 3] = _mm_loadu_si128((__m128i*)rightParentKey1);
				parentKeys[4 * w + 2] = _mm_loadu_si128((__m128i*)rightParentKey0);
			}

			//parentKeys[4 * w + 0] = _mm_loadu_si128((__m128i*)leftParentKey0);
			//parentKeys[4 * w + 1] = _mm_loadu_si128((__m128i*)leftParentKey1);
			//parentKeys[4 * w + 2] = _mm_loadu_si128((__m128i*)rightParentKey0);
			//parentKeys[4 * w + 3] = _mm_loadu_si128((__m128i*)rightParentKey1);

			parentKeys[4 * w + 0] = _mm_and_si128(signalBitCleaner, parentKeys[4 * w + 0]);
			parentKeys[4 * w + 1] = _mm_and_si128(signalBitCleaner, parentKeys[4 * w + 1]);
			parentKeys[4 * w + 2] = _mm_and_si128(signalBitCleaner, parentKeys[4 * w + 2]);
			parentKeys[4 * w + 3] = _mm_and_si128(signalBitCleaner, parentKeys[4 * w + 3]);

			/*
			std::cout << std::endl << std::endl;

			
			std::cout << "ParentKeys: " << std::endl;
			PrintKey(parentKeys[4 * w + 0]);
			std::cout << std::endl;
			PrintKey(parentKeys[4 * w + 1]);
			std::cout << std::endl;
			PrintKey(parentKeys[4 * w + 2]);
			std::cout << std::endl;
			PrintKey(parentKeys[4 * w + 3]);
			std::cout << std::endl;

			std::cout << "parent bits: " << (uint16_t)lpbit << (uint16_t)rpbit << std::endl;
			std::cout << "1 indicator: " << (uint16_t)combined_bits[w] << std::endl;
			*/
			


			data[8 * w + 0] = _mm_set_epi64x(0, m_vWireIds[currentGateIdx] + currentOffset); // K0 left
			data[8 * w + 1] = _mm_set_epi64x(1, m_vWireIds[currentGateIdx] + currentOffset); // K1 left
			data[8 * w + 2] = _mm_set_epi64x(2, m_vWireIds[currentGateIdx] + currentOffset); // K2 left
			data[8 * w + 3] = _mm_set_epi64x(3, m_vWireIds[currentGateIdx] + currentOffset); // K3 left
			data[8 * w + 4] = _mm_set_epi64x(0, m_vWireIds[currentGateIdx] + currentOffset); // K0 right
			data[8 * w + 5] = _mm_set_epi64x(2, m_vWireIds[currentGateIdx] + currentOffset); // K2 right
			data[8 * w + 6] = _mm_set_epi64x(1, m_vWireIds[currentGateIdx] + currentOffset); // K1 right
			data[8 * w + 7] = _mm_set_epi64x(3, m_vWireIds[currentGateIdx] + currentOffset); // K3 right

			targetGateKey0[w] = currentGate->gs.yinput.outKey[0] + 16 * currentOffset;
			targetGateKey1[w] = currentGate->gs.yinput.outKey[1] + 16 * currentOffset;

			currentOffset++;

			if (currentOffset >= currentGate->nvals)
			{
				currentGateIdx++;
				currentOffset = 0;
			}
		}

		aesni_encrypt_variable_keys<num_keys, 2>(parentKeys, data);

		for (size_t w = 0; w < width; ++w)
		{
			const __m128i key0 = _mm_xor_si128(data[8 * w + 0], data[8 * w + 4]);
			const __m128i key1 = _mm_xor_si128(data[8 * w + 1], data[8 * w + 6]);
			const __m128i key2 = _mm_xor_si128(data[8 * w + 2], data[8 * w + 5]);
			const __m128i key3 = _mm_xor_si128(data[8 * w + 3], data[8 * w + 7]);

			/*
			std::cout << "XOR Keys:" << std::endl;
			PrintKey(key0);
			std::cout << std::endl;
			PrintKey(key1);
			std::cout << std::endl;
			PrintKey(key2);
			std::cout << std::endl;
			PrintKey(key3);
			std::cout << std::endl;
			*/
			

			const __m128i tripleRedux = _mm_xor_si128(key3, _mm_xor_si128(key1, key2));
			if (combined_bits[w] != 0) {
				_mm_storeu_si128((__m128i*)targetGateKey0[w], key0);
				_mm_storeu_si128((__m128i*)targetGateKey1[w], tripleRedux);
			}
			else {
				_mm_storeu_si128((__m128i*)targetGateKey0[w], tripleRedux);
				_mm_storeu_si128((__m128i*)targetGateKey1[w], key0);
			}

			targetGateKey0[w][15] &= 0xFC;
			targetGateKey1[w][15] &= 0xFC;//0xFD;
			//targetGateKey1[w][15] ^= opbit[w];

			/*
			std::cout << "opbit: " << (uint16_t)opbit[w] << std::endl;

			std::cout << "Stored Key0: ";
			PrintKey(_mm_loadu_si128((__m128i*)targetGateKey0[w]));
			std::cout << std::endl;
			std::cout << "Stored Key1: ";
			PrintKey(_mm_loadu_si128((__m128i*)targetGateKey1[w]));
			std::cout << std::endl;
			*/

			uint8_t bit0 = (_mm_extract_epi8(key0, 15) & 0x01) ^ opbit[w];
			uint8_t bit1 = (_mm_extract_epi8(key1, 15) & 0x01) ^ opbit[w];
			uint8_t bit2 = (_mm_extract_epi8(key2, 15) & 0x01) ^ opbit[w];
			uint8_t bit3 = (_mm_extract_epi8(key3, 15) & 0x01) ^ opbit[w];

			switch (combined_bits[w])
			{
			case 0:
				bit0 ^= 1;
				break;
			case 1:
				bit1 ^= 1;
				break;
			case 2:
				bit2 ^= 1;
				break;
			case 3:
				bit3 ^= 1;
				break;
			default:
				break;
			}

			__m128i firstTable, secondTable;

			if (combined_bits[w] & 2) {
				firstTable = _mm_xor_si128(key0, key1);
			}
			else {
				firstTable = _mm_xor_si128(key2, key3);
			}
			if (combined_bits[w] & 1) {
				secondTable = _mm_xor_si128(key0, key2);
			}
			else {
				secondTable = _mm_xor_si128(key1, key3);
			}

			_mm_storeu_si128((__m128i*)gtptr, firstTable);
			gtptr[15] = (gtptr[15] & 0xFC) | bit0 | (bit1 << 1);
			gtptr += 16;
			_mm_storeu_si128((__m128i*)gtptr, secondTable);
			gtptr[15] = (gtptr[15] & 0xFC) | bit2 | (bit3 << 1);
			gtptr += 16;
		}
	}
}