#include "aesni_halfgate_processors.h"
#include "../yaosharing.h"
#include "aesni_helpers.h"

#include <wmmintrin.h>
#include <tmmintrin.h>
#include <emmintrin.h>
#include <immintrin.h>

#include <iostream>
#include <iomanip>
#include <algorithm>

// in number of tables
constexpr size_t mainGarblingWidthNI = 2;
constexpr size_t mainEvaluatingWidthNI = 4;

// ABY's default mixing strategy is a left shift followed by a XOR
// so we factored that out here
static inline __attribute__((always_inline)) __m128i aesni_mix_key_data(__m128i key, __m128i data) {
	// this is the 128-bit leftshift code from https://stackoverflow.com/a/34482688/4733946
	// as requested by User0 https://stackoverflow.com/users/5720018/user0
	// and given by Peter Cordes https://stackoverflow.com/users/224132/peter-cordes

	// 128-bit left-shift by 1
	__m128i carry = _mm_bslli_si128(key, 8);
	carry = _mm_srli_epi64(carry, 63);
	key = _mm_slli_epi64(key, 1);
	key = _mm_or_si128(key, carry);

	// then XOR in the data
	return _mm_xor_si128(data, key);
}


// a straight arbitrary width fixed-key encryption using AES-NI
template<size_t width>
static inline __attribute__((always_inline)) void aesni_encrypt_fixed_key(__m128i round_keys[11], __m128i data[width]) {
	__m128i whitening[width];

	for (size_t w = 0; w < width; ++w)
	{
		whitening[w] = data[w];
		data[w] = _mm_xor_si128(data[w], round_keys[0]);
	}

	for (size_t r = 1; r < 10; ++r)
	{
		for (size_t w = 0; w < width; ++w)
		{
			data[w] = _mm_aesenc_si128(data[w], round_keys[r]);
		}
	}

	for (size_t w = 0; w < width; ++w)
	{
		data[w] = _mm_aesenclast_si128(data[w], round_keys[10]);
		data[w] = _mm_xor_si128(data[w], whitening[w]);
	}

}

// anonymous namespace to facilitate inlining decisions
namespace {
	// helper struct to heavily reduce code duplication
	// across our three different Halfgate-based schemes
	// as all of them mostly differ in how exactly they mix the parent gate keys and the indices
	// both of these helpers represent exactly one AND garbling operation
	struct HalfGateGarbler
	{
	private:
		uint8_t rpbit;
		uint8_t lsbitANDrsbit;
		uint8_t* targetPiBit;
		uint8_t* targetGateKey;
		uint8_t* targetGateKeyR;
		__m128i rtable;

	public:
		// currentOffset is the current SIMD index value
		// the keys are the output
		inline __attribute__((always_inline)) void prepare(const GATE* currentGate,
			const std::vector<GATE>& vGates,
			uint32_t currentOffset,
			const __m128i R,
			__m128i keys[4]) {
			const uint32_t leftParentId = currentGate->ingates.inputs.twin.left;
			const uint32_t rightParentId = currentGate->ingates.inputs.twin.right;
			const GATE* leftParent = &vGates[leftParentId];
			const GATE* rightParent = &vGates[rightParentId];
			const uint8_t* leftParentKey = leftParent->gs.yinput.outKey[0] + 16 * currentOffset;
			const uint8_t* rightParentKey = rightParent->gs.yinput.outKey[0] + 16 * currentOffset;
			const uint8_t lpbit = leftParent->gs.yinput.pi[currentOffset];
			rpbit = rightParent->gs.yinput.pi[currentOffset];

			currentGate->gs.yinput.pi[currentOffset] = lpbit & rpbit;
			targetPiBit = currentGate->gs.yinput.pi + currentOffset;

			keys[0] = _mm_loadu_si128((__m128i*)leftParentKey);
			keys[1] = _mm_xor_si128(keys[0], R);
			keys[2] = _mm_loadu_si128((__m128i*)rightParentKey);
			keys[3] = _mm_xor_si128(keys[2], R);

			targetGateKey = currentGate->gs.yinput.outKey[0] + 16 * currentOffset;
			targetGateKeyR = currentGate->gs.yinput.outKey[1] + 16 * currentOffset;

			lsbitANDrsbit = (leftParentKey[15] & 0x01) & (rightParentKey[15] & 0x01);

			if (lpbit)
				rtable = keys[1];
			else
				rtable = keys[0];
		}

		// takes the AES results and writes the output into the garbled table
		inline __attribute__((always_inline)) void finalize(__m128i data[4],
			uint8_t* gtptr,
			const __m128i R) {
			__m128i ltable = _mm_xor_si128(data[0], data[1]);
			if (rpbit)
				ltable = _mm_xor_si128(ltable, R);
			_mm_storeu_si128((__m128i*)gtptr, ltable);
			gtptr += 16;
			const __m128i rXor = _mm_xor_si128(data[2], data[3]);
			rtable = _mm_xor_si128(rtable, rXor);
			_mm_storeu_si128((__m128i*)gtptr, rtable);
			gtptr += 16;

			__m128i outKey;
			
			if (rpbit)
				outKey = _mm_xor_si128(data[0], data[3]);
			else
				outKey = _mm_xor_si128(data[0], data[2]);
			if (lsbitANDrsbit)
				outKey = _mm_xor_si128(outKey, R);
			if (rpbit)
				outKey = _mm_xor_si128(outKey, rXor);
			/*
			if (rpbit)
				outKey = _mm_xor_si128(data[0], data[2]);
			else
				outKey = _mm_xor_si128(data[0], data[3]);
			if (lsbitANDrsbit)
				outKey = _mm_xor_si128(outKey, R);*/

			assert((_mm_extract_epi8(R, 15) & 0x01) == 1);
			uint8_t outWireBit = _mm_extract_epi8(outKey, 15) & 0x01;
			*targetPiBit ^= outWireBit;
			if (outWireBit)
			{
				outKey = _mm_xor_si128(outKey, R);
			}

			_mm_storeu_si128((__m128i*)targetGateKey, outKey);
			_mm_storeu_si128((__m128i*)targetGateKeyR, _mm_xor_si128(outKey, R));
		}
	};

	// this is the evaluation helper
	struct HalfGateEvaluator
	{
	private:
		__m128i finalMask;
		uint8_t* targetGateKey;
	public:
		// loads the garbled table for the current gate at its current SIMD value offset
		// and generates the output keys for AES processing
		inline __attribute__((always_inline)) void prepare(const GATE* currentGate,
			const std::vector<GATE>& vGates,
			uint32_t currentOffset,
			const uint8_t* gtptr,
			__m128i keys[2]) {
			const uint32_t leftParentId = currentGate->ingates.inputs.twin.left;
			const uint32_t rightParentId = currentGate->ingates.inputs.twin.right;
			const GATE* leftParent = &vGates[leftParentId];
			const GATE* rightParent = &vGates[rightParentId];
			const uint8_t* leftParentKey = leftParent->gs.yval + 16 * currentOffset;
			const uint8_t* rightParentKey = rightParent->gs.yval + 16 * currentOffset;

			keys[0] = _mm_loadu_si128((__m128i*)leftParentKey);
			keys[1] = _mm_loadu_si128((__m128i*)rightParentKey);

			targetGateKey = currentGate->gs.yval + 16 * currentOffset;

			// this is a value that will be XOR'ed in after the AES processing and only
			// depends on the input values and the garbled table, so we can preprocess and evict the other values from the registers
			// / caches
			finalMask = _mm_setzero_si128();
			if (leftParentKey[15] & 0x01)
			{
				finalMask = _mm_loadu_si128((__m128i*)gtptr);
			}
			gtptr += 16;
			if (rightParentKey[15] & 0x01)
			{
				__m128i temp = _mm_loadu_si128((__m128i*)gtptr);
				finalMask = _mm_xor_si128(finalMask, temp);
				finalMask = _mm_xor_si128(finalMask, keys[0]);
			}
			gtptr += 16;
		}

		inline __attribute__((always_inline)) void finalize(__m128i data[2]) {
			__m128i temp = _mm_xor_si128(data[0], data[1]);
			temp = _mm_xor_si128(temp, finalMask);
			_mm_storeu_si128((__m128i*)(targetGateKey), temp);
		}
	};
}


// width in number of tables
// bufferOffset in bytes
// fixedKey uses the gate counter mixed with the parent keys as the data input with a fixed AES key with post-whitening of the mixed value
template<size_t width>
void FixedKeyLTGarblingAesniProcessor::computeOutKeysAndTable(uint32_t tableCounter, size_t numTablesInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer)
{
	const __m128i ONE = _mm_set_epi32(0, 0, 0, 1);
	const __m128i TWO = _mm_set_epi32(0, 0, 0, 2);

	__m128i counter = _mm_set_epi32(0, 0, 0, tableCounter * KEYS_PER_GATE_IN_TABLE);

	// note: this implementation heavily relies on the fact that
	// the optimizer notices that all w-indexed loop iterations are independent
	// *and* that it manages to assign different registers to each iteration

	__m128i keys[4 * width];
	__m128i data[4 * width];
	__m128i aes_keys[11];

	for (size_t i = 0; i < 11; ++i)
	{
		aes_keys[i] = _mm_load_si128((__m128i*)(m_fixedKeyProvider.getExpandedStaticKey() + i * 16));
	}

	const __m128i R = _mm_loadu_si128((__m128i*)m_globalRandomOffset);

	uint32_t currentOffset = simdStartOffset;
	uint32_t currentGateIdx = queueStartIndex;
	uint8_t* gtptr = tableBuffer + 16 * KEYS_PER_GATE_IN_TABLE * tableCounter;

	for (size_t i = 0; i < numTablesInBatch; i += width)
	{
		HalfGateGarbler garblers[width];

		for (size_t w = 0; w < width; ++w)
		{
			const GATE* currentGate = m_tableGateQueue[currentGateIdx];

			garblers[w].prepare(currentGate, m_vGates, currentOffset, R, keys + 4 * w);

			currentOffset++;

			if (currentOffset >= currentGate->nvals)
			{
				currentGateIdx++;
				currentOffset = 0;
			}
		}

		for (size_t w = 0; w < width; ++w)
		{
			// use this because addition has a latency of 1 and a throughput of 0.5 CPI
			// also assignment should be free here if the compiler unrolls and assigns different registers
			data[4 * w + 0] = counter;
			data[4 * w + 1] = counter;
			data[4 * w + 2] = _mm_add_epi32(counter, ONE);
			data[4 * w + 3] = data[4 * w + 2];
			counter = _mm_add_epi32(counter, TWO);
		}

		for (size_t w = 0; w < 4 * width; ++w)
		{
			data[w] = aesni_mix_key_data(keys[w], data[w]);
		}

		aesni_encrypt_fixed_key<4 * width>(aes_keys, data);

		for (size_t w = 0; w < width; ++w)
		{
			garblers[w].finalize(data + 4 * w, gtptr, R);
			gtptr += 32;
		}
	}
}

void FixedKeyLTGarblingAesniProcessor::computeAESOutKeys(uint32_t tableCounter, size_t numTablesInBatch, uint8_t* tableBuffer)
{
	ProcessQueue(m_tableGateQueue, numTablesInBatch, tableCounter, tableBuffer);
}

size_t FixedKeyLTGarblingAesniProcessor::vectorWidth() const
{
	return mainGarblingWidthNI;
}

void FixedKeyLTGarblingAesniProcessor::BulkProcessor(uint32_t wireCounter, size_t numWiresInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer)
{
	computeOutKeysAndTable<mainGarblingWidthNI>(wireCounter, numWiresInBatch, queueStartIndex, simdStartOffset, tableBuffer);
}

void FixedKeyLTGarblingAesniProcessor::LeftoversProcessor(uint32_t wireCounter, size_t numWiresInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer)
{
	computeOutKeysAndTable<1>(wireCounter, numWiresInBatch, queueStartIndex, simdStartOffset, tableBuffer);
}

// width in number of tables
// bufferOffset in bytes
// InputKey uses the parent key as the actual AES key and the gate counter as the index
template<size_t width>
void InputKeyLTGarblingAesniProcessor::computeOutKeysAndTable(uint32_t tableCounter, size_t numTablesInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer)
{
	const __m128i ONE = _mm_set_epi32(0, 0, 0, 1);
	const __m128i TWO = _mm_set_epi32(0, 0, 0, 2);

	__m128i counter = _mm_set_epi32(0, 0, 0, tableCounter * KEYS_PER_GATE_IN_TABLE);

	// note: this implementation heavily relies on the fact that
	// the optimizer notices that all w-indexed loop iterations are independent
	// *and* that it manages to assign different registers to each iteration

	__m128i parentKeys[4 * width];
	__m128i data[4 * width];
	const __m128i R = _mm_loadu_si128((__m128i*)m_globalRandomOffset);

	uint32_t currentOffset = simdStartOffset;
	uint32_t currentGateIdx = queueStartIndex;
	uint8_t* gtptr = tableBuffer + 16 * KEYS_PER_GATE_IN_TABLE * tableCounter;

	for (size_t i = 0; i < numTablesInBatch; i += width)
	{
		HalfGateGarbler garblers[width];

		// TODO: optimize this to detect and exploit simd gates
		// saving vGate, parent base pointer and owning gate lookups
		for (size_t w = 0; w < width; ++w)
		{
			const GATE* currentGate = m_tableGateQueue[currentGateIdx];

			garblers[w].prepare(currentGate, m_vGates, currentOffset, R, parentKeys+4*w);

			currentOffset++;

			if (currentOffset >= currentGate->nvals)
			{
				currentGateIdx++;
				currentOffset = 0;
			}
		}

		for (size_t w = 0; w < width; ++w)
		{
			// use this because addition has a latency of 1 and a throughput of 0.5 CPI
			data[4 * w + 0] = counter;
			data[4 * w + 1] = counter;
			data[4 * w + 2] = _mm_add_epi32(counter, ONE);
			data[4 * w + 3] = data[4 * w + 2];
			counter = _mm_add_epi32(counter, TWO);
		}

		aesni_encrypt_variable_keys<4 * width,1>(parentKeys, data);

		for (size_t w = 0; w < width; ++w)
		{
			garblers[w].finalize(data + 4 * w, gtptr, R);
			gtptr += 32;
		}
	}
}

void InputKeyLTGarblingAesniProcessor::computeAESOutKeys(uint32_t tableCounter, size_t numTablesInBatch, uint8_t* tableBuffer)
{
	ProcessQueue(m_tableGateQueue, numTablesInBatch, tableCounter, tableBuffer);
}

size_t InputKeyLTGarblingAesniProcessor::vectorWidth() const
{
	return mainGarblingWidthNI;
}

void InputKeyLTGarblingAesniProcessor::BulkProcessor(uint32_t wireCounter, size_t numWiresInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer)
{
	computeOutKeysAndTable<mainGarblingWidthNI>(wireCounter, numWiresInBatch, queueStartIndex, simdStartOffset, tableBuffer);
}

void InputKeyLTGarblingAesniProcessor::LeftoversProcessor(uint32_t wireCounter, size_t numWiresInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer)
{
	computeOutKeysAndTable<1>(wireCounter, numWiresInBatch, queueStartIndex, simdStartOffset, tableBuffer);
}

static void expandAESKey(__m128i userkey, uint8_t* alignedStoragePointer)
{
	// this uses the fast AES key expansion (i.e. not using keygenassist) from Shay Gueron
	// https://eprint.iacr.org/2015/751

	__m128i temp1, temp2, temp3, globAux;
	const __m128i shuffle_mask =
		_mm_set_epi32(0x0c0f0e0d, 0x0c0f0e0d, 0x0c0f0e0d, 0x0c0f0e0d);
	const __m128i con3 = _mm_set_epi32(0x07060504, 0x07060504, 0x0ffffffff, 0x0ffffffff);
	__m128i rcon;
	temp1 = userkey;
	rcon = _mm_set_epi32(1, 1, 1, 1);
	_mm_store_si128((__m128i*)(alignedStoragePointer + 0 * 16), temp1);
	for (int i = 1; i <= 8; i++) {
		temp2 = _mm_shuffle_epi8(temp1, shuffle_mask);
		temp2 = _mm_aesenclast_si128(temp2, rcon);
		rcon = _mm_slli_epi32(rcon, 1);
		globAux = _mm_slli_epi64(temp1, 32);
		temp1 = _mm_xor_si128(globAux, temp1);
		globAux = _mm_shuffle_epi8(temp1, con3);
		temp1 = _mm_xor_si128(globAux, temp1);
		temp1 = _mm_xor_si128(temp2, temp1);
		_mm_store_si128((__m128i*)(alignedStoragePointer + i * 16), temp1);
	}
	rcon = _mm_set_epi32(0x1b, 0x1b, 0x1b, 0x1b);
	temp2 = _mm_shuffle_epi8(temp1, shuffle_mask);
	temp2 = _mm_aesenclast_si128(temp2, rcon);
	rcon = _mm_slli_epi32(rcon, 1);
	globAux = _mm_slli_epi64(temp1, 32);
	temp1 = _mm_xor_si128(globAux, temp1);
	globAux = _mm_shuffle_epi8(temp1, con3);
	temp1 = _mm_xor_si128(globAux, temp1);
	temp1 = _mm_xor_si128(temp2, temp1);
	_mm_store_si128((__m128i*)(alignedStoragePointer + 9 * 16), temp1);
	temp2 = _mm_shuffle_epi8(temp1, shuffle_mask);
	temp2 = _mm_aesenclast_si128(temp2, rcon);
	globAux = _mm_slli_epi64(temp1, 32);
	temp1 = _mm_xor_si128(globAux, temp1);
	globAux = _mm_shuffle_epi8(temp1, con3);
	temp1 = _mm_xor_si128(globAux, temp1);
	temp1 = _mm_xor_si128(temp2, temp1);
	_mm_store_si128((__m128i*)(alignedStoragePointer + 10 * 16), temp1);
}


template<size_t width>
inline void FixedKeyLTEvaluatingAesniProcessor::computeAESOutKeys(uint32_t tableCounter, size_t queueStartIndex, size_t simdStartOffset, size_t numTablesInBatch, const uint8_t* receivedTables)
{
	const __m128i ONE = _mm_set_epi32(0, 0, 0, 1);
	const __m128i TWO = _mm_set_epi32(0, 0, 0, 2);

	__m128i counter = _mm_set_epi32(0, 0, 0, tableCounter * KEYS_PER_GATE_IN_TABLE);

	// note: this implementation heavily relies on the fact that
	// the optimizer notices that all w-indexed loop iterations are independent
	// *and* that it manages to assign different registers to each iteration

	__m128i keys[2 * width];
	__m128i data[2 * width];
	__m128i aes_keys[11];
	const uint8_t* gtptr = receivedTables + tableCounter * KEYS_PER_GATE_IN_TABLE * 16;

	for (size_t i = 0; i < 11; ++i)
	{
		aes_keys[i] = _mm_load_si128((__m128i*)(m_fixedKeyProvider.getExpandedStaticKey() + i * 16));
	}

	size_t currentGateIdx = queueStartIndex;
	uint32_t currentOffset = simdStartOffset;

	for (size_t i = 0; i < numTablesInBatch; i += width)
	{
		HalfGateEvaluator evaluator[width];
		for (size_t w = 0; w < width; ++w)
		{
			const GATE* currentGate = m_gateQueue[currentGateIdx];
			evaluator[w].prepare(currentGate, m_vGates, currentOffset, gtptr, keys + 2 * w);
			gtptr += 32;

			currentOffset++;

			if (currentOffset >= currentGate->nvals)
			{
				currentGateIdx++;
				currentOffset = 0;
			}
		}

		for (size_t w = 0; w < width; ++w)
		{
			// use this because addition has a latency of 1 and a throughput of 0.5 CPI
			data[2 * w + 0] = counter;
			data[2 * w + 1] = _mm_add_epi32(counter, ONE);
			counter = _mm_add_epi32(counter, TWO);
		}

		for (size_t w = 0; w < 2 * width; ++w)
		{
			data[w] = aesni_mix_key_data(keys[w], data[w]);
		}

		aesni_encrypt_fixed_key<2 * width>(aes_keys, data);

		for (size_t w = 0; w < width; ++w)
		{
			evaluator[w].finalize(data + 2 * w);
		}
	}
}


template<size_t width>
inline void InputKeyLTEvaluatingAesniProcessor::computeAESOutKeys(uint32_t tableCounter, size_t queueStartIndex, size_t simdStartOffset, size_t numTablesInBatch, const uint8_t* receivedTables)
{
	const __m128i ONE = _mm_set_epi32(0, 0, 0, 1);
	const __m128i TWO = _mm_set_epi32(0, 0, 0, 2);

	__m128i counter = _mm_set_epi32(0, 0, 0, tableCounter * KEYS_PER_GATE_IN_TABLE);

	// note: this implementation heavily relies on the fact that
	// the optimizer notices that all w-indexed loop iterations are independent
	// *and* that it manages to assign different registers to each iteration

	__m128i parentKeys[2 * width];
	__m128i data[2 * width];
	const uint8_t* gtptr = receivedTables + tableCounter * KEYS_PER_GATE_IN_TABLE * 16;


	size_t currentGateIdx = queueStartIndex;
	uint32_t currentOffset = simdStartOffset;

	for (size_t i = 0; i < numTablesInBatch; i += width)
	{
		HalfGateEvaluator evaluator[width];

		for (size_t w = 0; w < width; ++w)
		{
			const GATE* currentGate = m_gateQueue[currentGateIdx];
			evaluator[w].prepare(currentGate, m_vGates, currentOffset, gtptr, parentKeys + 2 * w);
			gtptr += 32;

			currentOffset++;

			if (currentOffset >= currentGate->nvals)
			{
				currentGateIdx++;
				currentOffset = 0;
			}
		}

		for (size_t w = 0; w < width; ++w)
		{
			// use this because addition has a latency of 1 and a throughput of 0.5 CPI
			data[2 * w + 0] = counter;
			data[2 * w + 1] = _mm_add_epi32(counter, ONE);
			counter = _mm_add_epi32(counter, TWO);
		}

		aesni_encrypt_variable_keys<2 * width,1>(parentKeys, data);

		for (size_t w = 0; w < width; ++w)
		{
			evaluator[w].finalize(data + 2 * w);
		}
	}
}

void FixedKeyLTEvaluatingAesniProcessor::computeAESOutKeys(uint32_t tableCounter, size_t numTablesInBatch, uint8_t* receivedTables)
{
	ProcessQueue(m_gateQueue, numTablesInBatch, tableCounter, receivedTables);
}

size_t FixedKeyLTEvaluatingAesniProcessor::vectorWidth() const
{
	return mainEvaluatingWidthNI;
}

void FixedKeyLTEvaluatingAesniProcessor::BulkProcessor(uint32_t wireCounter, size_t numWiresInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer)
{
	computeAESOutKeys<mainEvaluatingWidthNI>(wireCounter, queueStartIndex, simdStartOffset, numWiresInBatch, tableBuffer);
}

void FixedKeyLTEvaluatingAesniProcessor::LeftoversProcessor(uint32_t wireCounter, size_t numWiresInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer)
{
	computeAESOutKeys<1>(wireCounter, queueStartIndex, simdStartOffset, numWiresInBatch, tableBuffer);
}

void InputKeyLTEvaluatingAesniProcessor::computeAESOutKeys(uint32_t tableCounter, size_t numTablesInBatch, uint8_t* receivedTables)
{
	ProcessQueue(m_gateQueue, numTablesInBatch, tableCounter, receivedTables);
}

size_t InputKeyLTEvaluatingAesniProcessor::vectorWidth() const
{
	return mainEvaluatingWidthNI;
}

void InputKeyLTEvaluatingAesniProcessor::BulkProcessor(uint32_t wireCounter, size_t numWiresInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer)
{
	computeAESOutKeys<mainEvaluatingWidthNI>(wireCounter, queueStartIndex, simdStartOffset, numWiresInBatch, tableBuffer);
}

void InputKeyLTEvaluatingAesniProcessor::LeftoversProcessor(uint32_t wireCounter, size_t numWiresInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer)
{
	computeAESOutKeys<1>(wireCounter, queueStartIndex, simdStartOffset, numWiresInBatch, tableBuffer);
}

void FixedKeyProvider::expandAESKey(const uint8_t* userkey)
{
	if (userkey) {
		__m128i key = _mm_loadu_si128((__m128i*)userkey);
		::expandAESKey(key, m_expandedStaticAESKey.get());
	}
	else {
		// note that order is most significant to least significant byte for this intrinsic
		const __m128i key = _mm_set_epi8(0xFF, 0xEE, 0xDD, 0xCC, 0xBB, 0xAA, 0x99, 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11, 0x00);
		::expandAESKey(key, m_expandedStaticAESKey.get());
	}
}

void MILTGarblingAesniProcessor::computeAESOutKeys(uint32_t tableCounter, size_t numTablesInBatch, uint8_t* tableBuffer)
{
	ProcessQueue(m_tableGateQueue, numTablesInBatch, tableCounter, tableBuffer);
}

size_t MILTGarblingAesniProcessor::vectorWidth() const
{
	return mainGarblingWidthNI;
}

void MILTGarblingAesniProcessor::BulkProcessor(uint32_t wireCounter, size_t numWiresInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer)
{
	computeOutKeysAndTable<mainGarblingWidthNI>(wireCounter, numWiresInBatch, queueStartIndex, simdStartOffset, tableBuffer);
}

void MILTGarblingAesniProcessor::LeftoversProcessor(uint32_t wireCounter, size_t numWiresInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer)
{
	computeOutKeysAndTable<1>(wireCounter, numWiresInBatch, queueStartIndex, simdStartOffset, tableBuffer);
}

void MILTEvaluatingAesniProcessor::computeAESOutKeys(uint32_t tableCounter, size_t numTablesInBatch, uint8_t* receivedTables)
{
	ProcessQueue(m_gateQueue, numTablesInBatch, tableCounter, receivedTables);
}

size_t MILTEvaluatingAesniProcessor::vectorWidth() const
{
	return mainEvaluatingWidthNI;
}

void MILTEvaluatingAesniProcessor::BulkProcessor(uint32_t wireCounter, size_t numWiresInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer)
{
	computeAESOutKeys<mainEvaluatingWidthNI>(wireCounter, queueStartIndex, simdStartOffset, numWiresInBatch, tableBuffer);
}

void MILTEvaluatingAesniProcessor::LeftoversProcessor(uint32_t wireCounter, size_t numWiresInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer)
{
	computeAESOutKeys<1>(wireCounter, queueStartIndex, simdStartOffset, numWiresInBatch, tableBuffer);
}

// this uses the specific orthomorphism from the paper https://eprint.iacr.org/2019/1168
static inline __attribute__((always_inline)) __m128i aesni_orthomorphism(__m128i value) {
	const __m128i extractor = _mm_set_epi64x(~0,0);
	__m128i swapped = _mm_shuffle_epi32(value, 78);
	__m128i extracted = _mm_and_si128(value, extractor);
	return _mm_xor_si128(swapped, extracted);
}


// this uses the index as the AES key and the parent key as the data input after applying the above orthmorphism
// and then using post-whitening of the input value
template<size_t width>
void MILTGarblingAesniProcessor::computeOutKeysAndTable(uint32_t tableCounter, size_t numTablesInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer)
{
	const __m128i ONE = _mm_set_epi32(0, 0, 0, 1);
	const __m128i TWO = _mm_set_epi32(0, 0, 0, 2);


	__m128i counter = _mm_set_epi32(0, 0, 0, tableCounter * KEYS_PER_GATE_IN_TABLE);
	const __m128i idxOffset = _mm_loadu_si128((__m128i*)m_globalIndexOffset);
	counter = _mm_add_epi32(counter, idxOffset);

	// note: this implementation heavily relies on the fact that
	// the optimizer notices that all w-indexed loop iterations are independent
	// *and* that it manages to assign different registers to each iteration

	__m128i parentKeys[4 * width];
	__m128i data[2 * width];
	__m128i whiteningValues[4 * width];
	const __m128i R = _mm_loadu_si128((__m128i*)m_globalRandomOffset);

	uint32_t currentOffset = simdStartOffset;
	uint32_t currentGateIdx = queueStartIndex;
	uint8_t* gtptr = tableBuffer + 16 * KEYS_PER_GATE_IN_TABLE * tableCounter;

	for (size_t i = 0; i < numTablesInBatch; i += width)
	{
		HalfGateGarbler garblers[width];

		// TODO: optimize this to detect and exploit simd gates
		// saving vGate, parent base pointer and owning gate lookups
		for (size_t w = 0; w < width; ++w)
		{
			const GATE* currentGate = m_tableGateQueue[currentGateIdx];

			garblers[w].prepare(currentGate, m_vGates, currentOffset, R, parentKeys + 4 * w);

			currentOffset++;

			if (currentOffset >= currentGate->nvals)
			{
				currentGateIdx++;
				currentOffset = 0;
			}
		}

		for (size_t w = 0; w < width; ++w)
		{
			// use this because addition has a latency of 1 and a throughput of 0.5 CPI
			data[2 * w + 0] = counter;
			data[2 * w + 1] = _mm_add_epi32(counter, ONE);
			counter = _mm_add_epi32(counter, TWO);
		}

		for (size_t w = 0; w < 4 * width; ++w) {
			parentKeys[w] = aesni_orthomorphism(parentKeys[w]);
			whiteningValues[w] = parentKeys[w];
		}

		aesni_encrypt_variable_keys<2 * width, 2>(data, parentKeys);

		for (size_t w = 0; w < 4 * width; ++w) {
			parentKeys[w] = _mm_xor_si128(parentKeys[w], whiteningValues[w]);
		}

		for (size_t w = 0; w < width; ++w)
		{
			garblers[w].finalize(parentKeys + 4 * w, gtptr, R);
			gtptr += 32;
		}
	}
}

template<size_t width>
inline void MILTEvaluatingAesniProcessor::computeAESOutKeys(uint32_t tableCounter, size_t queueStartIndex, size_t simdStartOffset, size_t numTablesInBatch, const uint8_t* receivedTables)
{
	const __m128i ONE = _mm_set_epi32(0, 0, 0, 1);
	const __m128i TWO = _mm_set_epi32(0, 0, 0, 2);

	__m128i counter = _mm_set_epi32(0, 0, 0, tableCounter * KEYS_PER_GATE_IN_TABLE);
	const __m128i idxOffset = _mm_loadu_si128((__m128i*)m_globalIndexOffset);
	PrintKey(idxOffset);
	std::cout << std::endl;
	counter = _mm_add_epi32(counter, idxOffset);

	// note: this implementation heavily relies on the fact that
	// the optimizer notices that all w-indexed loop iterations are independent
	// *and* that it manages to assign different registers to each iteration

	__m128i parentKeys[2 * width];
	__m128i data[2 * width];
	__m128i whiteningValues[2 * width];
	const uint8_t* gtptr = receivedTables + tableCounter * KEYS_PER_GATE_IN_TABLE * 16;


	size_t currentGateIdx = queueStartIndex;
	uint32_t currentOffset = simdStartOffset;

	for (size_t i = 0; i < numTablesInBatch; i += width)
	{
		HalfGateEvaluator evaluator[width];

		for (size_t w = 0; w < width; ++w)
		{
			const GATE* currentGate = m_gateQueue[currentGateIdx];
			evaluator[w].prepare(currentGate, m_vGates, currentOffset, gtptr, parentKeys + 2 * w);
			gtptr += 32;

			currentOffset++;

			if (currentOffset >= currentGate->nvals)
			{
				currentGateIdx++;
				currentOffset = 0;
			}
		}

		for (size_t w = 0; w < width; ++w)
		{
			// use this because addition has a latency of 1 and a throughput of 0.5 CPI
			data[2 * w + 0] = counter;
			data[2 * w + 1] = _mm_add_epi32(counter, ONE);
			counter = _mm_add_epi32(counter, TWO);
		}

		for (size_t w = 0; w < 2 * width; ++w) {
			parentKeys[w] = aesni_orthomorphism(parentKeys[w]);
			whiteningValues[w] = parentKeys[w];
		}

		aesni_encrypt_variable_keys<2 * width, 1>(data, parentKeys);

		for (size_t w = 0; w < 2 * width; ++w) {
			parentKeys[w] = _mm_xor_si128(parentKeys[w], whiteningValues[w]);
		}

		for (size_t w = 0; w < width; ++w)
		{
			evaluator[w].finalize(parentKeys + 2 * w);
		}
	}
}