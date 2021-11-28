#include "vaes_prf_processors.h"
#include "vaes_helpers.h"
#include "aesni_helpers.h"

#include <cassert>

#include <wmmintrin.h>
#include <tmmintrin.h>
#include <smmintrin.h>
#include <emmintrin.h>
#include <immintrin.h>


constexpr size_t mainEvaluatingWidthXor = 16;
constexpr size_t mainEvaluatingWidthAnd = 16;
constexpr size_t mainGarblingWidthAnd = 6; // 4
constexpr size_t mainGarblingWidthXor = 12;

void PRFXorLTEvaluatingVaesProcessor::computeAESOutKeys(uint32_t tableCounter, size_t numTablesInBatch, uint8_t* receivedTables)
{
	ProcessQueue(m_gateQueue, numTablesInBatch, tableCounter, receivedTables);
}

size_t PRFXorLTEvaluatingVaesProcessor::vectorWidth() const
{
	return mainEvaluatingWidthXor;
}

void PRFXorLTEvaluatingVaesProcessor::BulkProcessor(uint32_t wireCounter, size_t numWiresInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer)
{
	computeAESOutKeys<mainEvaluatingWidthXor>(wireCounter, queueStartIndex, simdStartOffset, numWiresInBatch, tableBuffer);
}

void PRFXorLTEvaluatingVaesProcessor::LeftoversProcessor(uint32_t wireCounter, size_t numWiresInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer)
{
	computeAESOutKeys<1>(wireCounter, queueStartIndex, simdStartOffset, numWiresInBatch, tableBuffer);
}

// unlike the halfgates garbling operations
// we always stick to a minimal gate parallelism of 4 here
// giving each gate its own lane within a register and then using one register for each value

template<size_t width>
void PRFXorLTEvaluatingVaesProcessor::computeAESOutKeys(uint32_t tableCounter, size_t queueStartIndex, size_t simdStartOffset, size_t numTablesInBatch, const uint8_t* receivedTables)
{
	constexpr size_t num_words = (width + 3) / 4;
	constexpr size_t num_buffer_words = std::min(width, size_t(4));

	static_assert(width < 4 || width % 4 == 0, "This implementation only works for small widths or fitting widths");

	// note: this implementation heavily relies on the fact that
	// the optimizer notices that all w-indexed loop iterations are independent
	// *and* that it manages to assign different registers to each iteration

	__m512i leftData[num_words];
	__m512i leftKeys[num_words];
	__m512i rightData[num_words];
	__m512i rightKeys[num_words];
	__m512i finalMask[num_words];
	__mmask8 doRXor[num_words];
	__m512i opbit[num_words];
	uint8_t* targetGateKey[width];
	const uint8_t* gtptr = receivedTables + tableCounter * 16;

	const __m128i signalBitCleaner = _mm_set_epi8(0xFE, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
	const __m128i permBitCleaner = _mm_set_epi8(0x01, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
	const __m512i wideCleaner = _mm512_broadcast_i32x4(signalBitCleaner);
	const __m512i widePermCleaner = _mm512_broadcast_i32x4(permBitCleaner);


	size_t currentGateIdx = queueStartIndex;
	uint32_t currentOffset = simdStartOffset;

	// one big loop for the progression
	for (size_t i = 0; i < numTablesInBatch; i += width)
	{
		// one loop for all ZMM registers
		for (size_t w = 0; w < num_words; ++w) {
			doRXor[w] = 0;

//#pragma unroll
			// one loop to fill a ZMM register
			for (size_t k = 0; k < num_buffer_words; ++k) {
				const GATE* currentGate = m_gateQueue[currentGateIdx];
				const uint32_t leftParentId = currentGate->ingates.inputs.twin.left;
				const uint32_t rightParentId = currentGate->ingates.inputs.twin.right;
				const GATE* leftParent = &m_vGates[leftParentId];
				const GATE* rightParent = &m_vGates[rightParentId];
				const uint8_t* leftParentKey = leftParent->gs.yval + 16 * currentOffset;
				const uint8_t* rightParentKey = rightParent->gs.yval + 16 * currentOffset;

				const __m128i leftKey = _mm_loadu_si128((__m128i*)leftParentKey);
				leftKeys[w] = mm512_insert_128(leftKeys[w], leftKey,k);
				const __m128i rightKey = _mm_loadu_si128((__m128i*)rightParentKey);
				rightKeys[w] = mm512_insert_128(rightKeys[w], rightKey, k);

				const uint8_t lpbit = leftParentKey[15] & 0x01;
				const uint8_t rpbit = rightParentKey[15] & 0x01;
				const uint8_t expandedRpbit = (rpbit << 1) | rpbit;
				doRXor[w] |= expandedRpbit << (2 * k);

				const __m128i leftLocalData = _mm_set_epi64x(lpbit, m_vWireIds[currentGateIdx] + currentOffset);
				leftData[w] = mm512_insert_128(leftData[w], leftLocalData, k);
				const __m128i rightLocalData = _mm_set_epi64x(rpbit, m_vWireIds[currentGateIdx] + currentOffset);
				rightData[w] = mm512_insert_128(rightData[w], rightLocalData, k);

				targetGateKey[4 * w + k] = currentGate->gs.yval + 16 * currentOffset;

				// we can't always use the below wide-vector load because the upper parts of the loaded chunk may segfault on the last table entry
				// but if we have a width that is a multiple of 4, we're guaranteed to have at least 4 tables to load
				// we use an if constexpr here to guarantee not to pay any runtime cost for this optimization
				if constexpr (width % 4 != 0) {
					const __m128i finalLocalMask = _mm_loadu_si128((__m128i*)gtptr); // resolution happens later in a blend operation
					gtptr += 16;
					finalMask[w] = mm512_insert_128(finalMask[w], finalLocalMask, k);
				}

				currentOffset++;

				if (currentOffset >= currentGate->nvals)
				{
					currentGateIdx++;
					currentOffset = 0;
				}
			}

			// do the generic preprocessing

			opbit[w] = _mm512_xor_si512(leftKeys[w],rightKeys[w]);
			leftKeys[w] = _mm512_and_si512(leftKeys[w], wideCleaner);
			rightKeys[w] = _mm512_and_si512(rightKeys[w], wideCleaner);

			if constexpr (width % 4 == 0) {
				finalMask[w] = _mm512_mask_loadu_epi64(rightKeys[w], doRXor[w], (__m512i*)gtptr);
				gtptr += 64;
			}
			else { // clean up the insertions all at once
				finalMask[w] = _mm512_mask_blend_epi64(doRXor[w], rightKeys[w], finalMask[w]);
			}
			
			opbit[w] = _mm512_and_si512(opbit[w], widePermCleaner);
		}

		vaes_encrypt_variable_keys(num_words, leftKeys, leftData, rightKeys, rightData);

		for (size_t w = 0; w < num_words; ++w)
		{
			__m512i temp = _mm512_xor_si512(leftData[w], finalMask[w]);
			temp = _mm512_mask_xor_epi64(temp, doRXor[w], temp, rightData[w]);
			temp = _mm512_and_si512(temp, wideCleaner);
			temp = _mm512_or_si512(temp, opbit[w]);

//#pragma unroll
			for (size_t k = 0; k < num_buffer_words; ++k) {
				__m128i store_word = mm512_extract_128(temp, k);
				_mm_storeu_si128((__m128i*)(targetGateKey[4*w+k]), store_word);
			}
		}
	}
}

void PRFXorLTGarblingVaesProcessor::computeAESOutKeys(uint32_t tableCounter, size_t numTablesInBatch, uint8_t* receivedTables)
{
	ProcessQueue(m_gateQueue, numTablesInBatch, tableCounter, receivedTables);
}

size_t PRFXorLTGarblingVaesProcessor::vectorWidth() const
{
	return mainGarblingWidthXor;
}

void PRFXorLTGarblingVaesProcessor::BulkProcessor(uint32_t wireCounter, size_t numWiresInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer)
{
	computeAESOutKeys<mainGarblingWidthXor>(wireCounter, queueStartIndex, simdStartOffset, numWiresInBatch, tableBuffer);
}

void PRFXorLTGarblingVaesProcessor::LeftoversProcessor(uint32_t wireCounter, size_t numWiresInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer)
{
	computeAESOutKeys<1>(wireCounter, queueStartIndex, simdStartOffset, numWiresInBatch, tableBuffer);
}

template<size_t width>
void PRFXorLTGarblingVaesProcessor::computeAESOutKeys(uint32_t tableCounter, size_t queueStartIndex, size_t simdStartOffset, size_t numTablesInBatch, uint8_t* receivedTables)
{
	constexpr size_t num_words = (width + 3) / 4;
	constexpr size_t num_buffer_words = std::min(width, size_t(4));

	static_assert(width < 4 || width % 4 == 0, "This implementation only works for small widths or fitting widths");

	// note: this implementation heavily relies on the fact that
	// the optimizer notices that all w-indexed loop iterations are independent
	// *and* that it manages to assign different registers to each iteration

	__m512i leftDeltaData[num_words];
	__m512i rightDeltaData[num_words];
	__m512i rightPRFData[num_words];

	__m512i leftDeltaKey[num_words];
	__m512i rightDeltaKey[num_words];
	__m512i rightPRFKey[num_words];

	__m512i rightBypassData[num_words];

	uint8_t* targetGateKey0[width];
	uint8_t* targetGateKey1[width];
	__mmask8 rpbitMask[num_words];
	uint8_t* gtptr = receivedTables + tableCounter * 16;
	const __m128i signalBitCleanerSmall = _mm_set_epi8(0xFE, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
	const __m512i signalBitCleaner = _mm512_broadcast_i32x4(signalBitCleanerSmall);

	size_t currentGateIdx = queueStartIndex;
	uint32_t currentOffset = simdStartOffset;

	for (size_t i = 0; i < numTablesInBatch; i += width)
	{
		for (size_t w = 0; w < num_words; ++w)
		{
			rpbitMask[w] = 0;
			__m512i rightKey0;
			__m512i rightKey1;
//#pragma unroll
			for (size_t k = 0; k < num_buffer_words; ++k) {
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
				const uint8_t localRpBit = rightParent->gs.yinput.pi[currentOffset];
				currentGate->gs.yinput.pi[currentOffset] = lpbit ^ localRpBit;
				rpbitMask[w] |= ((localRpBit << 1) | localRpBit) << (2*k);

				const __m128i leftDeltaKeyLocal = _mm_loadu_si128((__m128i*)leftParentKey0);
				leftDeltaKey[w] = mm512_insert_128(leftDeltaKey[w], leftDeltaKeyLocal, k);
				const __m128i rightDeltaKeyLocal = _mm_loadu_si128((__m128i*)leftParentKey1);
				rightDeltaKey[w] = mm512_insert_128(rightDeltaKey[w], rightDeltaKeyLocal, k);

				const __m128i rightKey0Local = _mm_loadu_si128((__m128i*)rightParentKey0);
				rightKey0 = mm512_insert_128(rightKey0, rightKey0Local, k);
				const __m128i rightKey1Local = _mm_loadu_si128((__m128i*)rightParentKey1);
				rightKey1 = mm512_insert_128(rightKey1, rightKey1Local, k);

				const __m128i leftDeltaDataLocal = _mm_set_epi64x(lpbit, m_vWireIds[currentGateIdx] + currentOffset);
				leftDeltaData[w] = mm512_insert_128(leftDeltaData[w], leftDeltaDataLocal, k);
				const __m128i rightDeltaDataLocal = _mm_set_epi64x(1 - lpbit, m_vWireIds[currentGateIdx] + currentOffset);
				rightDeltaData[w] = mm512_insert_128(rightDeltaData[w], rightDeltaDataLocal, k);
				const __m128i rightPRFDataLocal = _mm_set_epi64x(1, m_vWireIds[currentGateIdx] + currentOffset);
				rightPRFData[w] = mm512_insert_128(rightPRFData[w], rightPRFDataLocal, k);

				targetGateKey0[4 * w + k] = currentGate->gs.yinput.outKey[0] + 16 * currentOffset;
				targetGateKey1[4 * w + k] = currentGate->gs.yinput.outKey[1] + 16 * currentOffset;

				currentOffset++;

				if (currentOffset >= currentGate->nvals)
				{
					currentGateIdx++;
					currentOffset = 0;
				}
			}

			rightBypassData[w] = _mm512_mask_blend_epi64(rpbitMask[w], rightKey0, rightKey1);
			rightPRFKey[w] = _mm512_mask_blend_epi64(rpbitMask[w], rightKey1, rightKey0);

			leftDeltaKey[w] = _mm512_and_si512(leftDeltaKey[w], signalBitCleaner);
			rightDeltaKey[w] = _mm512_and_si512(rightDeltaKey[w], signalBitCleaner);
			rightPRFKey[w] = _mm512_and_si512(rightPRFKey[w], signalBitCleaner);
			rightBypassData[w] = _mm512_and_si512(rightBypassData[w], signalBitCleaner);

		}

		vaes_encrypt_variable_keys(num_words, leftDeltaKey, leftDeltaData, rightDeltaKey, rightDeltaData, rightPRFKey, rightPRFData);

		for (size_t w = 0; w < num_words; ++w)
		{
			const __m512i delta = _mm512_xor_si512(leftDeltaData[w], rightDeltaData[w]);

			const __m512i tableMask = _mm512_xor_si512(rightBypassData[w], delta);
			const __m512i tableEntry = _mm512_xor_si512(rightPRFData[w], tableMask);
			const __m512i rtkey0 = _mm512_mask_blend_epi64(rpbitMask[w], rightBypassData[w], tableMask);

			if constexpr (width % 4 == 0) {
				_mm512_storeu_si512((__m512i*)gtptr, tableEntry);
				gtptr += 64;
			}
			else {
//#pragma unroll
				for (size_t k = 0; k < num_buffer_words; ++k) {
					const __m128i localEntry = mm512_extract_128(tableEntry, k);
					_mm_storeu_si128((__m128i*)gtptr, localEntry);
					gtptr += 16;
				}
			}
			
			const __m512i outkey0 = _mm512_xor_si512(leftDeltaData[w], rtkey0);
			const __m512i outkey1 = _mm512_xor_si512(rightDeltaData[w], rtkey0);

//#pragma unroll
			for (size_t k = 0; k < num_buffer_words; ++k) {
				const __m128i localKey0 = mm512_extract_128(outkey0, k);
				_mm_storeu_si128((__m128i*)targetGateKey0[4*w+k], localKey0);
				const __m128i localKey1 = mm512_extract_128(outkey1, k);
				_mm_storeu_si128((__m128i*)targetGateKey1[4 * w + k], localKey1);
			}
		}
	}
}


void PRFAndLTEvaluatingVaesProcessor::computeAESOutKeys(uint32_t tableCounter, size_t numTablesInBatch, uint8_t* receivedTables)
{
	ProcessQueue(m_gateQueue, numTablesInBatch, tableCounter, receivedTables);
}

size_t PRFAndLTEvaluatingVaesProcessor::vectorWidth() const
{
	return mainEvaluatingWidthAnd;
}

void PRFAndLTEvaluatingVaesProcessor::BulkProcessor(uint32_t wireCounter, size_t numWiresInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer)
{
	computeAESOutKeys<mainEvaluatingWidthAnd>(wireCounter, queueStartIndex, simdStartOffset, numWiresInBatch, tableBuffer);
}

void PRFAndLTEvaluatingVaesProcessor::LeftoversProcessor(uint32_t wireCounter, size_t numWiresInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer)
{
	computeAESOutKeys<1>(wireCounter, queueStartIndex, simdStartOffset, numWiresInBatch, tableBuffer);
}

template<size_t width>
void PRFAndLTEvaluatingVaesProcessor::computeAESOutKeys(uint32_t tableCounter, size_t queueStartIndex, size_t simdStartOffset, size_t numTablesInBatch, const uint8_t* receivedTables)
{
	constexpr size_t num_words = (width + 3) / 4;
	constexpr size_t num_buffer_words = std::min(width, size_t(4));

	static_assert(width < 4 || width % 4 == 0, "This implementation only works for small widths or fitting widths");

	// note: this implementation heavily relies on the fact that
	// the optimizer notices that all w-indexed loop iterations are independent
	// *and* that it manages to assign different registers to each iteration

	__m512i leftData[num_words];
	__m512i rightData[num_words];
	__m512i leftKeys[num_words];
	__m512i rightKeys[num_words];
	__m512i finalMask[num_words];
	uint8_t* targetGateKey[width];
	__m512i opbit[num_words];
	const uint8_t* gtptr = receivedTables + tableCounter * 2 * 16;
	const __m128i signalBitCleanerSmall = _mm_set_epi8(0xFE, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
	const __m128i tableBitsCleanerSmall = _mm_set_epi8(0xFC, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
	const __m128i permBitExtractorSmall = _mm_set_epi8(0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00);
	const __m128i nonSignalTablebitCleanerSmall = _mm_set_epi8(0xFD, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);

	const __m512i signalBitCleaner = _mm512_broadcast_i32x4(signalBitCleanerSmall);
	const __m512i tableBitsCleaner = _mm512_broadcast_i32x4(tableBitsCleanerSmall);
	const __m512i permBitExtractor = _mm512_broadcast_i32x4(permBitExtractorSmall);
	const __m512i nonSignalTablebitCleaner = _mm512_broadcast_i32x4(nonSignalTablebitCleanerSmall);


	size_t currentGateIdx = queueStartIndex;
	uint32_t currentOffset = simdStartOffset;

	for (size_t i = 0; i < numTablesInBatch; i += width)
	{
		for (size_t w = 0; w < num_words; ++w)
		{
//#pragma unroll
			for (size_t k = 0; k < num_buffer_words; ++k) {
				const GATE* currentGate = m_gateQueue[currentGateIdx];
				const uint32_t leftParentId = currentGate->ingates.inputs.twin.left;
				const uint32_t rightParentId = currentGate->ingates.inputs.twin.right;
				const GATE* leftParent = &m_vGates[leftParentId];
				const GATE* rightParent = &m_vGates[rightParentId];
				const uint8_t* leftParentKey = leftParent->gs.yval + 16 * currentOffset;
				const uint8_t* rightParentKey = rightParent->gs.yval + 16 * currentOffset;

				const __m128i leftKey = _mm_loadu_si128((__m128i*)leftParentKey);
				leftKeys[w] = mm512_insert_128(leftKeys[w], leftKey, k);
				const __m128i rightKey = _mm_loadu_si128((__m128i*)rightParentKey);
				rightKeys[w] = mm512_insert_128(rightKeys[w], rightKey, k);

				const uint8_t lpbit = leftParentKey[15] & 0x01;
				const uint8_t rpbit = rightParentKey[15] & 0x01;
				const uint8_t rpbit11 = (rpbit << 1) | rpbit;
				const uint8_t lpbit11 = (lpbit << 1) | lpbit;
				const uint8_t combined_bits = (lpbit << 1) | rpbit;

				const __m128i leftDataLocal = _mm_set_epi64x(combined_bits, m_vWireIds[currentGateIdx] + currentOffset);
				leftData[w] = mm512_insert_128(leftData[w], leftDataLocal, k);
				// right data is a straight copy

				targetGateKey[4*w+k] = currentGate->gs.yval + 16 * currentOffset;

				__m128i firstTable = _mm_loadu_si128((__m128i*)gtptr);
				const uint8_t transmittedBitFirst = _mm_extract_epi8(firstTable,15) & 0x03;
				__m128i finalMaskLocal = _mm_mask_blend_epi64(rpbit11, _mm_setzero_si128(), firstTable);
				gtptr += 16;
				__m128i secondTable = _mm_loadu_si128((__m128i*)gtptr);
				const uint8_t transmittedBitSecond = _mm_extract_epi8(secondTable,15) & 0x03;
				finalMaskLocal = _mm_mask_xor_epi64(finalMaskLocal, lpbit11, finalMaskLocal, secondTable);
				gtptr += 16;

				finalMask[w] = mm512_insert_128(finalMask[w], finalMaskLocal, k);

				const uint8_t joinedBits = (transmittedBitSecond << 2) | transmittedBitFirst;
				const uint8_t opbitLocal = (joinedBits >> combined_bits) & 0x01;

				opbit[w] = mm512_insert_128(opbit[w], _mm_set1_epi8(opbitLocal), k);


				currentOffset++;

				if (currentOffset >= currentGate->nvals)
				{
					currentGateIdx++;
					currentOffset = 0;
				}
			}	

			leftKeys[w] = _mm512_and_si512(leftKeys[w], signalBitCleaner);
			rightKeys[w] = _mm512_and_si512(rightKeys[w], signalBitCleaner);
			rightData[w] = leftData[w];
			finalMask[w] = _mm512_and_si512(finalMask[w], tableBitsCleaner);
			opbit[w] = _mm512_and_si512(opbit[w], permBitExtractor);
		}

		vaes_encrypt_variable_keys(num_words, leftKeys, leftData, rightKeys, rightData);

		for (size_t w = 0; w < num_words; ++w)
		{
			__m512i temp = _mm512_xor_si512(leftData[w], rightData[w]);
			temp = _mm512_and_si512(nonSignalTablebitCleaner, temp);
			temp = _mm512_xor_si512(temp, finalMask[w]);
			temp = _mm512_xor_si512(temp, opbit[w]);
			for (size_t k = 0; k < num_buffer_words; ++k) {
				__m128i extracted = mm512_extract_128(temp, k);
				_mm_storeu_si128((__m128i*)(targetGateKey[4*w+k]), extracted);
			}
		}
	}
}

void PRFAndLTGarblingVaesProcessor::computeAESOutKeys(uint32_t tableCounter, size_t numTablesInBatch, uint8_t* receivedTables)
{
	ProcessQueue(m_gateQueue, numTablesInBatch, tableCounter, receivedTables);
}

size_t PRFAndLTGarblingVaesProcessor::vectorWidth() const
{
	return mainGarblingWidthAnd;
}

void PRFAndLTGarblingVaesProcessor::BulkProcessor(uint32_t wireCounter, size_t numWiresInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer)
{
	computeAESOutKeys<mainGarblingWidthAnd>(wireCounter, queueStartIndex, simdStartOffset, numWiresInBatch, tableBuffer);
}

void PRFAndLTGarblingVaesProcessor::LeftoversProcessor(uint32_t wireCounter, size_t numWiresInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer)
{
	computeAESOutKeys<1>(wireCounter, queueStartIndex, simdStartOffset, numWiresInBatch, tableBuffer);
}


template<size_t width>
void PRFAndLTGarblingVaesProcessor::computeAESOutKeys(uint32_t tableCounter, size_t queueStartIndex, size_t simdStartOffset, size_t numTablesInBatch, uint8_t* receivedTables)
{
	constexpr size_t num_data = 2 * width;

	// note: this implementation heavily relies on the fact that
	// the optimizer notices that all w-indexed loop iterations are independent
	// *and* that it manages to assign different registers to each iteration

	__m512i data[num_data];
	__m512i parentKeys[width]; // lpi, l!pi, rpi, r!pi
	uint8_t* targetGateKey0[width];
	uint8_t* targetGateKey1[width];
	uint8_t combined_bits[width];
	__m512i opbit[width];
	uint8_t* gtptr = receivedTables + tableCounter * 2 * 16;
	const __m128i signalBitCleanerSmall = _mm_set_epi8(0xFE, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
	const __m512i signalBitCleaner = _mm512_broadcast_i32x4(signalBitCleanerSmall);
	const __m128i tableBitsCleanerSmall = _mm_set_epi8(0xFC, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
	const __m512i tableBitsCleaner = _mm512_broadcast_i32x4(tableBitsCleanerSmall);
	const __m128i permBitExtractorSmall = _mm_set_epi8(0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00);
	const __m512i permBitExtractor = _mm512_broadcast_i32x4(permBitExtractorSmall);

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
			opbit[w] = _mm512_set1_epi8(currentGate->gs.yinput.pi[currentOffset]);
			//std::cout << "pi: " << (uint16_t)currentGate->gs.yinput.pi[currentOffset] << std::endl;
			combined_bits[w] = ((1 - lpbit) << 1) | (1 - rpbit);

			const uint8_t lpbit11 = (lpbit << 1) | lpbit;
			const uint8_t rpbit11 = (rpbit << 1) | rpbit;

			__m128i loadedLeftKey0 = _mm_loadu_si128((__m128i*)leftParentKey0);
			__m128i loadedLeftKey1 = _mm_loadu_si128((__m128i*)leftParentKey1);

			__m128i lowerLeftTemp = _mm_mask_blend_epi64(lpbit11, loadedLeftKey0, loadedLeftKey1);
			__m128i upperLeftTemp = _mm_mask_blend_epi64(lpbit11, loadedLeftKey1, loadedLeftKey0);
			parentKeys[w] = _mm512_castsi128_si512(lowerLeftTemp);
			parentKeys[w] = _mm512_inserti32x4(parentKeys[w], upperLeftTemp, 1);

			__m128i loadedRightKey0 = _mm_loadu_si128((__m128i*)rightParentKey0);
			__m128i loadedRightKey1 = _mm_loadu_si128((__m128i*)rightParentKey1);

			__m128i lowerRightTemp = _mm_mask_blend_epi64(rpbit11, loadedRightKey0, loadedRightKey1);
			__m128i upperRightTemp = _mm_mask_blend_epi64(rpbit11, loadedRightKey1, loadedRightKey0);
			__m256i rightTemp = _mm256_castsi128_si256(lowerRightTemp);

			rightTemp = _mm256_inserti128_si256(rightTemp, upperRightTemp, 1);
			parentKeys[w] = _mm512_inserti64x4(parentKeys[w], rightTemp, 1);

			/*data[8 * w + 0] = _mm_set_epi64x(0, m_vWireIds[currentGateIdx] + currentOffset); // K0 left
			data[8 * w + 1] = _mm_set_epi64x(1, m_vWireIds[currentGateIdx] + currentOffset); // K1 left
			data[8 * w + 2] = _mm_set_epi64x(2, m_vWireIds[currentGateIdx] + currentOffset); // K2 left
			data[8 * w + 3] = _mm_set_epi64x(3, m_vWireIds[currentGateIdx] + currentOffset); // K3 left */

			const uint64_t global_gate_id = m_vWireIds[currentGateIdx] + currentOffset;

			/*data[8 * w + 4] = _mm_set_epi64x(0, m_vWireIds[currentGateIdx] + currentOffset); // K0 right
			data[8 * w + 5] = _mm_set_epi64x(2, m_vWireIds[currentGateIdx] + currentOffset); // K2 right
			data[8 * w + 6] = _mm_set_epi64x(1, m_vWireIds[currentGateIdx] + currentOffset); // K1 right
			data[8 * w + 7] = _mm_set_epi64x(3, m_vWireIds[currentGateIdx] + currentOffset); // K3 right*/

			// previously it would always encrypt in pairs, i.e. (0,1),(2,3),...
			// which now means we need to put these index pairs into the same lane on our pair of 512-bit registers

			data[2 * w + 0] = _mm512_set_epi64(1, global_gate_id, 0, global_gate_id, 2, global_gate_id, 0, global_gate_id);
			data[2 * w + 1] = _mm512_set_epi64(3, global_gate_id, 2, global_gate_id, 3, global_gate_id, 1, global_gate_id);

			parentKeys[w] = _mm512_and_si512(parentKeys[w], signalBitCleaner);

			targetGateKey0[w] = currentGate->gs.yinput.outKey[0] + 16 * currentOffset;
			targetGateKey1[w] = currentGate->gs.yinput.outKey[1] + 16 * currentOffset;

			currentOffset++;

			if (currentOffset >= currentGate->nvals)
			{
				currentGateIdx++;
				currentOffset = 0;
			}
		}

		vaes_encrypt_variable_keys_variable_data<width, 2>(parentKeys, data);

		for (size_t w = 0; w < width; ++w)
		{
			// bit 4 decides whether left or right input word is taken
			// bits 3-1 decide which word from a vector is taken
			const __m512i leftShuffle = _mm512_set_epi64(
				3,2,
				11,10,
				1,0,
				9,8
			);
			const __m512i rightShuffle = _mm512_set_epi64(
				7,6,
				5,4,
				15,14,
				13,12
			);
			const __m512i shuffledLeft = _mm512_permutex2var_epi64(data[2 * w + 1], leftShuffle, data[2 * w + 0]);
			const __m512i shuffledRight = _mm512_permutex2var_epi64(data[2 * w + 1], rightShuffle, data[2 * w + 0]);
			
			const __m512i fourKeys = _mm512_xor_si512(shuffledLeft, shuffledRight);

			//PrintKey(fourKeys);
			//std::cout << std::endl;

			/*
			const __m128i key0 = _mm_xor_si128(data[8 * w + 0], data[8 * w + 4]);
			const __m128i key1 = _mm_xor_si128(data[8 * w + 1], data[8 * w + 6]);
			const __m128i key2 = _mm_xor_si128(data[8 * w + 2], data[8 * w + 5]);
			const __m128i key3 = _mm_xor_si128(data[8 * w + 3], data[8 * w + 7]);
			*/

			const __m512i tableCleanedKeys = _mm512_and_si512(fourKeys, tableBitsCleaner);
			const __m512i permutedBits = _mm512_xor_si512(fourKeys,opbit[w]);
			//const __m512i preparedBits = _mm512_and_si512(permutedBits, permBitExtractor);

			//std::cout << "cleaned bits: " << std::endl;
			//PrintKey(preparedBits);

			const __m128i key0 = _mm512_extracti32x4_epi32(tableCleanedKeys, 0);
			const __m128i key1 = _mm512_extracti32x4_epi32(tableCleanedKeys, 1);
			const __m128i key2 = _mm512_extracti32x4_epi32(tableCleanedKeys, 2);
			const __m128i key3 = _mm512_extracti32x4_epi32(tableCleanedKeys, 3);


			const __m128i tripleRedux = _mm_xor_si128(key3, _mm_xor_si128(key1, key2));
			const uint8_t storeAmplifiedCombinedBits = combined_bits[w] | (combined_bits[w] << 1) | (combined_bits[w] >> 1);
			const __m128i zeroKey = _mm_mask_blend_epi64(storeAmplifiedCombinedBits, tripleRedux, key0);
			const __m128i oneKey = _mm_mask_blend_epi64(storeAmplifiedCombinedBits, key0, tripleRedux);
			_mm_storeu_si128((__m128i*)targetGateKey0[w], zeroKey);
			_mm_storeu_si128((__m128i*)targetGateKey1[w], oneKey);
			/*if (combined_bits[w] != 0) {
				_mm_storeu_si128((__m128i*)targetGateKey0[w], key0);
				_mm_storeu_si128((__m128i*)targetGateKey1[w], tripleRedux);
			}
			else {
				_mm_storeu_si128((__m128i*)targetGateKey0[w], tripleRedux);
				_mm_storeu_si128((__m128i*)targetGateKey1[w], key0);
			}*/

			/*std::cout << "Stored Key0: ";
			PrintKey(_mm_loadu_si128((__m128i*)targetGateKey0[w]));
			std::cout << std::endl;
			std::cout << "Stored Key1: ";
			PrintKey(_mm_loadu_si128((__m128i*)targetGateKey1[w]));
			std::cout << std::endl;*/

			const __m512i extraction_indices = _mm512_set1_epi8(56);
			// this could be replaced with a masking AND and a following comparison instruction
			// however, there should be plenty of time to hide the high latency bitshuffle
			uint64_t extracted_bits = _mm512_mask_bitshuffle_epi64_mask(0x01'00'01'00'01'00'01'00, permutedBits, extraction_indices);

			//std::cout << "extracted bits: " << extracted_bits << std::endl;

			extracted_bits ^= (1ull << (combined_bits[w] * 16 + 8));

			__m128i firstTable, secondTable;

			const uint8_t firstTableBit = combined_bits[w] & 2;
			const uint8_t firstTableBit11 = firstTableBit | (firstTableBit >> 1);
			const uint8_t secondTableBit = combined_bits[w] & 1;
			const uint8_t secondTableBit11 = secondTableBit | (secondTableBit << 1);

			firstTable = _mm_xor_si128(key2, key3);
			firstTable = _mm_mask_xor_epi64(firstTable, firstTableBit11, key0, key1);

			secondTable = _mm_xor_si128(key1, key3);
			secondTable = _mm_mask_xor_epi64(secondTable, secondTableBit11, key0, key2);

			/*if (combined_bits[w] & 2) {
				firstTable = _mm_xor_si128(key0, key1);
			}
			else {
				firstTable = _mm_xor_si128(key2, key3);
			}*/
			/*if (combined_bits[w] & 1) {
				secondTable = _mm_xor_si128(key0, key2);
			}
			else {
				secondTable = _mm_xor_si128(key1, key3);
			}*/

			_mm_storeu_si128((__m128i*)gtptr, firstTable);
			gtptr[15] |= ((extracted_bits >> 8) | ((extracted_bits >> 24) <<1)) & 0x03;
			gtptr += 16;
			_mm_storeu_si128((__m128i*)gtptr, secondTable);
			gtptr[15] |= ((extracted_bits >> 40) | ((extracted_bits >> 56)<<1)) & 0x03;
			gtptr += 16;
		}
	}
}