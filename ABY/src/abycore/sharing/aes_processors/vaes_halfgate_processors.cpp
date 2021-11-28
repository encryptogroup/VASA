#include "vaes_halfgate_processors.h"
#include "../yaosharing.h"
#include "vaes_helpers.h"

#include <wmmintrin.h>
#include <tmmintrin.h>
#include <emmintrin.h>
#include <immintrin.h>

#include <iostream>
#include <iomanip>
#include <algorithm>

// in number of tables
constexpr size_t mainGarblingWidthPRPVaes = 8;
constexpr size_t mainEvaluatingWidthPRPVaes = 16;
constexpr size_t mainGarblingWidthCircVaes = 6;
constexpr size_t mainEvaluatingWidthCircVaes = 16;
constexpr size_t mainGarblingWidthMIVaes = 6;
constexpr size_t mainEvaluatingWidthMIVaes = 16;

// ABY's left-shift algorithm using vector instructions to do 4x128 vectors
static inline __attribute__((always_inline)) __m512i vaes_mix_keys(__m512i key, __m512i data) {
	// this is the 128-bit leftshift code from https://stackoverflow.com/a/34482688/4733946
	// as requested by User0 https://stackoverflow.com/users/5720018/user0
	// and given by Peter Cordes https://stackoverflow.com/users/224132/peter-cordes

	__m512i carry = _mm512_bslli_epi128(key, 8);
	carry = _mm512_srli_epi64(carry, 63);
	key = _mm512_slli_epi64(key, 1);
	key = _mm512_or_si512(key, carry);

	return _mm512_xor_si512(data, key);
}

// The orthomorphism for multi-instance garbling for 4x128 vector applications
static inline __attribute__((always_inline)) __m512i vaes_orthomorphism(__m512i value) {
	const __m128i extractor_small = _mm_set_epi64x(~0, 0);
	const __m512i extractor = _mm512_broadcast_i32x4(extractor_small);
	__m512i swapped = _mm512_shuffle_epi32(value, _MM_PERM_BADC);
	__m512i extracted = _mm512_and_si512(value, extractor);
	return _mm512_xor_si512(swapped, extracted);
}

void FixedKeyLTEvaluatingVaesProcessor::computeAESOutKeys(uint32_t tableCounter, size_t numTablesInBatch, uint8_t* receivedTables)
{
	ProcessQueue(m_gateQueue, numTablesInBatch, tableCounter, receivedTables);
}

size_t FixedKeyLTEvaluatingVaesProcessor::vectorWidth() const
{
	return mainEvaluatingWidthPRPVaes;
}

void FixedKeyLTEvaluatingVaesProcessor::BulkProcessor(uint32_t wireCounter, size_t numWiresInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer)
{
	computeAESOutKeys<mainEvaluatingWidthPRPVaes>(wireCounter, queueStartIndex, simdStartOffset, numWiresInBatch, tableBuffer);
}

void FixedKeyLTEvaluatingVaesProcessor::LeftoversProcessor(uint32_t wireCounter, size_t numWiresInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer)
{
	computeAESOutKeys<1>(wireCounter, queueStartIndex, simdStartOffset, numWiresInBatch, tableBuffer);
}

void FixedKeyLTGarblingVaesProcessor::computeAESOutKeys(uint32_t tableCounter, size_t numTablesInBatch, uint8_t* tableBuffer)
{
	ProcessQueue(m_tableGateQueue, numTablesInBatch, tableCounter, tableBuffer);
}

size_t FixedKeyLTGarblingVaesProcessor::vectorWidth() const
{
	return mainGarblingWidthPRPVaes;
}

void FixedKeyLTGarblingVaesProcessor::BulkProcessor(uint32_t wireCounter, size_t numWiresInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer)
{
	computeOutKeysAndTable<mainGarblingWidthPRPVaes>(wireCounter, numWiresInBatch, queueStartIndex, simdStartOffset, tableBuffer);
}

void FixedKeyLTGarblingVaesProcessor::LeftoversProcessor(uint32_t wireCounter, size_t numWiresInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer)
{
	computeOutKeysAndTable<1>(wireCounter, numWiresInBatch, queueStartIndex, simdStartOffset, tableBuffer);
}


void InputKeyLTEvaluatingVaesProcessor::computeAESOutKeys(uint32_t tableCounter, size_t numTablesInBatch, uint8_t* receivedTables)
{
	ProcessQueue(m_gateQueue, numTablesInBatch, tableCounter, receivedTables);
}

size_t InputKeyLTEvaluatingVaesProcessor::vectorWidth() const
{
	return mainEvaluatingWidthCircVaes;
}

void InputKeyLTEvaluatingVaesProcessor::BulkProcessor(uint32_t wireCounter, size_t numWiresInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer)
{
	computeAESOutKeys<mainEvaluatingWidthCircVaes>(wireCounter, queueStartIndex, simdStartOffset, numWiresInBatch, tableBuffer);
}

void InputKeyLTEvaluatingVaesProcessor::LeftoversProcessor(uint32_t wireCounter, size_t numWiresInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer)
{
	computeAESOutKeys<1>(wireCounter, queueStartIndex, simdStartOffset, numWiresInBatch, tableBuffer);
}

void InputKeyLTGarblingVaesProcessor::computeAESOutKeys(uint32_t tableCounter, size_t numTablesInBatch, uint8_t* tableBuffer)
{
	ProcessQueue(m_tableGateQueue, numTablesInBatch, tableCounter, tableBuffer);
}

size_t InputKeyLTGarblingVaesProcessor::vectorWidth() const
{
	return mainGarblingWidthCircVaes;
}

void InputKeyLTGarblingVaesProcessor::BulkProcessor(uint32_t wireCounter, size_t numWiresInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer)
{
	computeOutKeysAndTable<mainGarblingWidthCircVaes>(wireCounter, numWiresInBatch, queueStartIndex, simdStartOffset, tableBuffer);
}

void InputKeyLTGarblingVaesProcessor::LeftoversProcessor(uint32_t wireCounter, size_t numWiresInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer)
{
	computeOutKeysAndTable<1>(wireCounter, numWiresInBatch, queueStartIndex, simdStartOffset, tableBuffer);
}


void MILTEvaluatingVaesProcessor::computeAESOutKeys(uint32_t tableCounter, size_t numTablesInBatch, uint8_t* receivedTables)
{
	ProcessQueue(m_gateQueue, numTablesInBatch, tableCounter, receivedTables);
}

size_t MILTEvaluatingVaesProcessor::vectorWidth() const
{
	return mainEvaluatingWidthMIVaes;
}

void MILTEvaluatingVaesProcessor::BulkProcessor(uint32_t wireCounter, size_t numWiresInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer)
{
	computeAESOutKeys<mainEvaluatingWidthMIVaes>(wireCounter, queueStartIndex, simdStartOffset, numWiresInBatch, tableBuffer);
}

void MILTEvaluatingVaesProcessor::LeftoversProcessor(uint32_t wireCounter, size_t numWiresInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer)
{
	computeAESOutKeys<1>(wireCounter, queueStartIndex, simdStartOffset, numWiresInBatch, tableBuffer);
}

void MILTGarblingVaesProcessor::computeAESOutKeys(uint32_t tableCounter, size_t numTablesInBatch, uint8_t* tableBuffer)
{
	ProcessQueue(m_tableGateQueue, numTablesInBatch, tableCounter, tableBuffer);
}

size_t MILTGarblingVaesProcessor::vectorWidth() const
{
	return mainGarblingWidthMIVaes;
}

void MILTGarblingVaesProcessor::BulkProcessor(uint32_t wireCounter, size_t numWiresInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer)
{
	computeOutKeysAndTable<mainGarblingWidthMIVaes>(wireCounter, numWiresInBatch, queueStartIndex, simdStartOffset, tableBuffer);
}

void MILTGarblingVaesProcessor::LeftoversProcessor(uint32_t wireCounter, size_t numWiresInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer)
{
	computeOutKeysAndTable<1>(wireCounter, numWiresInBatch, queueStartIndex, simdStartOffset, tableBuffer);
}

// these implementations work by assigning each 128-bit lane out of the four available
// to one garbling operation

template<size_t width>
inline void FixedKeyLTEvaluatingVaesProcessor::computeAESOutKeys(uint32_t tableCounter, size_t queueStartIndex, size_t simdStartOffset, size_t numTablesInBatch, const uint8_t* receivedTables)
{
	constexpr size_t div_width = (width + 3) / 4; // ceiling division
	constexpr size_t num_buffer_words = std::min(width, size_t(4));

	static_assert((width < 4) || (width % 4 == 0));

	// we need this because the baseline difference counter between the lanes is 2
	const __m512i ONE = _mm512_set_epi32(
		0, 0, 0, 1,
		0, 0, 0, 1,
		0, 0, 0, 1,
		0, 0, 0, 1);

	// the offset calculation WILL break down if the above static assert is violated
	// Example: width==7
	// then the first 4 would need an offset of 8
	// and then the second 3 would need an offset of 6 so the next round has the right offset
	// we *could* fix this with a div_width array of offsets where the first div_width-1 elements have the 8
	// and the remainder has (width%4)*2 however we don't need such weird instances
	constexpr size_t offset = std::min(size_t(4) * KEYS_PER_GATE_IN_TABLE, width * KEYS_PER_GATE_IN_TABLE);

	// we need to advance all counter values by a full cycle after each gate
	// so that gate 1 and 3 will not accidentally share a counter value
	const __m512i FULL_OFFSET = _mm512_set_epi32(
		0, 0, 0, offset,
		0, 0, 0, offset,
		0, 0, 0, offset,
		0, 0, 0, offset);



	__m512i counter = _mm512_set_epi32(
		0, 0, 0, (tableCounter + 3) * KEYS_PER_GATE_IN_TABLE,
		0, 0, 0, (tableCounter + 2) * KEYS_PER_GATE_IN_TABLE,
		0, 0, 0, (tableCounter + 1) * KEYS_PER_GATE_IN_TABLE,
		0, 0, 0, (tableCounter + 0) * KEYS_PER_GATE_IN_TABLE);

	// note: this implementation heavily relies on the fact that
	// the optimizer notices that all w-indexed loop iterations are independent
	// *and* that it manages to assign different registers to each iteration

	__m512i leftData[div_width];
	__m512i rightData[div_width];
	__m512i leftKeys[div_width];
	__m512i rightKeys[div_width];
	__m512i finalMask[div_width];
	uint8_t* targetGateKey[width];
	__m512i aes_keys[11];
	const uint8_t* gtptr = receivedTables + tableCounter * KEYS_PER_GATE_IN_TABLE * 16;

	for (size_t i = 0; i < 11; ++i)
	{
		__m128i temp_key = _mm_load_si128((__m128i*)(m_fixedKeyProvider.getExpandedStaticKey() + i * 16));
		aes_keys[i] = _mm512_broadcast_i32x4(temp_key);
	}

	size_t currentGateIdx = queueStartIndex;
	uint32_t currentOffset = simdStartOffset;

	for (size_t i = 0; i < numTablesInBatch; i += width)
	{
		// TODO: optimize using bigger vector loads potentially?
		for (size_t w = 0; w < div_width; ++w)
		{
			for (size_t k = 0; k < num_buffer_words; ++k) {
				const GATE* currentGate = m_gateQueue[currentGateIdx];
				const uint32_t leftParentId = currentGate->ingates.inputs.twin.left;
				const uint32_t rightParentId = currentGate->ingates.inputs.twin.right;
				const GATE* leftParent = &m_vGates[leftParentId];
				const GATE* rightParent = &m_vGates[rightParentId];
				const uint8_t* leftParentKey = leftParent->gs.yval + 16 * currentOffset;
				const uint8_t* rightParentKey = rightParent->gs.yval + 16 * currentOffset;

				const __m128i leftParentKeyLocal = _mm_loadu_si128((__m128i*)leftParentKey);
				leftKeys[w] = mm512_insert_128(leftKeys[w], leftParentKeyLocal, k);
				const __m128i rightParentKeyLocal = _mm_loadu_si128((__m128i*)rightParentKey);
				rightKeys[w] = mm512_insert_128(rightKeys[w], rightParentKeyLocal, k);

				targetGateKey[4 * w + k] = currentGate->gs.yval + 16 * currentOffset;

				const uint8_t lpbit = leftParentKey[15] & 0x01;
				const uint8_t lpbit11 = (lpbit << 1) | lpbit;
				const uint8_t rpbit = rightParentKey[15] & 0x01;
				const uint8_t rpbit11 = (rpbit << 1) | rpbit;

				// this is a conditionally zeroed load based upon the value of the left permutation bit
				__m128i finalMaskLocal = _mm_maskz_loadu_epi64(lpbit11, (__m128i*)gtptr);
				gtptr += 16;
				const __m128i rightTable = _mm_loadu_si128((__m128i*)gtptr);
				const __m128i rightMaskUpdate = _mm_xor_si128(rightTable, leftParentKeyLocal);
				// this saves us the use of an if to conditionally apply the preprocessing mask
				finalMaskLocal = _mm_mask_xor_epi64(finalMaskLocal, rpbit11, finalMaskLocal, rightMaskUpdate);
				gtptr += 16;

				// this invokes a switch-case that the compiler can hopefully turn into an explicit insertion
				finalMask[w] = mm512_insert_128(finalMask[w], finalMaskLocal, k);

				currentOffset++;

				if (currentOffset >= currentGate->nvals)
				{
					currentGateIdx++;
					currentOffset = 0;
				}
			}
		}

		for (size_t w = 0; w < div_width; ++w)
		{
			// use this because addition has a latency of 1 and a throughput of 0.5 CPI
			leftData[w] = counter;
			rightData[w] = _mm512_add_epi32(counter, ONE);
			counter = _mm512_add_epi32(counter, FULL_OFFSET);
		}

		for (size_t w = 0; w < div_width; ++w)
		{
			leftData[w] = vaes_mix_keys(leftKeys[w], leftData[w]);
			rightData[w] = vaes_mix_keys(rightKeys[w], rightData[w]);

			leftKeys[w] = leftData[w]; // keep as a backup for post-whitening
			rightKeys[w] = rightData[w]; // keep as a backup for post-whitening
		}

		vaes_encrypt_fixed_keys(div_width, aes_keys, leftData, rightData);

		for (size_t w = 0; w < div_width; ++w)
		{
			// post-whitening and value combination
			leftData[w] = _mm512_xor_si512(leftData[w], leftKeys[w]);
			rightData[w] = _mm512_xor_si512(rightData[w], rightKeys[w]);
			leftData[w] = _mm512_xor_si512(leftData[w], rightData[w]);
			leftData[w] = _mm512_xor_si512(leftData[w], finalMask[w]);
		}

		for (size_t w = 0; w < div_width; ++w)
		{
			for (size_t k = 0; k < num_buffer_words; ++k) {
				const __m128i extracted = mm512_extract_128(leftData[w], k);
				_mm_storeu_si128((__m128i*)(targetGateKey[4 * w + k]), extracted);
			}
		}
	}
}

// this is largely analogous to the FixedKey version
template<size_t width>
inline void InputKeyLTEvaluatingVaesProcessor::computeAESOutKeys(uint32_t tableCounter, size_t queueStartIndex, size_t simdStartOffset, size_t numTablesInBatch, const uint8_t* receivedTables)
{
	constexpr size_t div_width = (width + 3) / 4; // ceiling division
	constexpr size_t num_buffer_words = std::min(width, size_t(4));

	static_assert((width < 4) || (width % 4 == 0), "This implementation only supports multiplies of 4 or values smaller than 4.");

	const __m512i ONE = _mm512_set_epi32(
		0, 0, 0, 1,
		0, 0, 0, 1,
		0, 0, 0, 1,
		0, 0, 0, 1);

	// the offset calculation WILL break down if the above static assert is violated
	// Example: width==7
	// then the first 4 would need an offset of 8
	// and then the second 3 would need an offset of 6 so the next round has the right offset
	// we *could* fix this with a div_width array of offsets where the first div_width-1 elements have the 8
	// and the remainder has (width%4)*2 however we don't need such weird instances
	constexpr size_t offset = std::min(size_t(4) * KEYS_PER_GATE_IN_TABLE, width * KEYS_PER_GATE_IN_TABLE);

	const __m512i FULL_OFFSET = _mm512_set_epi32(
		0, 0, 0, offset,
		0, 0, 0, offset,
		0, 0, 0, offset,
		0, 0, 0, offset);



	__m512i counter = _mm512_set_epi32(
		0, 0, 0, (tableCounter + 3) * KEYS_PER_GATE_IN_TABLE,
		0, 0, 0, (tableCounter + 2) * KEYS_PER_GATE_IN_TABLE,
		0, 0, 0, (tableCounter + 1) * KEYS_PER_GATE_IN_TABLE,
		0, 0, 0, (tableCounter + 0) * KEYS_PER_GATE_IN_TABLE);

	// note: this implementation heavily relies on the fact that
	// the optimizer notices that all w-indexed loop iterations are independent
	// *and* that it manages to assign different registers to each iteration

	__m512i leftData[div_width];
	__m512i rightData[div_width];
	__m512i leftKeys[div_width];
	__m512i rightKeys[div_width];
	__m512i finalMask[div_width];
	uint8_t* targetGateKey[width];
	const uint8_t* gtptr = receivedTables + tableCounter * KEYS_PER_GATE_IN_TABLE * 16;

	size_t currentGateIdx = queueStartIndex;
	uint32_t currentOffset = simdStartOffset;

	for (size_t i = 0; i < numTablesInBatch; i += width)
	{
		// TODO: optimize using bigger vector loads potentially?
		for (size_t w = 0; w < div_width; ++w)
		{
			for (size_t k = 0; k < num_buffer_words; ++k) {
				const GATE* currentGate = m_gateQueue[currentGateIdx];
				const uint32_t leftParentId = currentGate->ingates.inputs.twin.left;
				const uint32_t rightParentId = currentGate->ingates.inputs.twin.right;
				const GATE* leftParent = &m_vGates[leftParentId];
				const GATE* rightParent = &m_vGates[rightParentId];
				const uint8_t* leftParentKey = leftParent->gs.yval + 16 * currentOffset;
				const uint8_t* rightParentKey = rightParent->gs.yval + 16 * currentOffset;

				const __m128i leftParentKeyLocal = _mm_loadu_si128((__m128i*)leftParentKey);
				leftKeys[w] = mm512_insert_128(leftKeys[w], leftParentKeyLocal, k);
				const __m128i rightParentKeyLocal = _mm_loadu_si128((__m128i*)rightParentKey);
				rightKeys[w] = mm512_insert_128(rightKeys[w], rightParentKeyLocal, k);

				targetGateKey[4 * w + k] = currentGate->gs.yval + 16 * currentOffset;

				const uint8_t lpbit = leftParentKey[15] & 0x01;
				const uint8_t lpbit11 = (lpbit << 1) | lpbit;
				const uint8_t rpbit = rightParentKey[15] & 0x01;
				const uint8_t rpbit11 = (rpbit << 1) | rpbit;

				__m128i finalMaskLocal = _mm_maskz_loadu_epi64(lpbit11, (__m128i*)gtptr);
				gtptr += 16;
				const __m128i rightTable = _mm_loadu_si128((__m128i*)gtptr);
				const __m128i rightMaskUpdate = _mm_xor_si128(rightTable, leftParentKeyLocal);
				finalMaskLocal = _mm_mask_xor_epi64(finalMaskLocal, rpbit11, finalMaskLocal, rightMaskUpdate);
				gtptr += 16;

				finalMask[w] = mm512_insert_128(finalMask[w], finalMaskLocal, k);

				currentOffset++;

				if (currentOffset >= currentGate->nvals)
				{
					currentGateIdx++;
					currentOffset = 0;
				}
			}
		}

		for (size_t w = 0; w < div_width; ++w)
		{
			// use this because addition has a latency of 1 and a throughput of 0.5 CPI
			leftData[w] = counter;
			rightData[w] = _mm512_add_epi32(counter, ONE);
			counter = _mm512_add_epi32(counter, FULL_OFFSET);

		}

		vaes_encrypt_variable_keys(div_width, leftKeys, leftData, rightKeys, rightData);

		for (size_t w = 0; w < div_width; ++w)
		{
			leftData[w] = _mm512_xor_si512(leftData[w], rightData[w]);
			leftData[w] = _mm512_xor_si512(leftData[w], finalMask[w]);
		}

		for (size_t w = 0; w < div_width; ++w)
		{
			for (size_t k = 0; k < num_buffer_words; ++k) {
				const __m128i extracted = mm512_extract_128(leftData[w], k);
				_mm_storeu_si128((__m128i*)(targetGateKey[4 * w + k]), extracted);
			}
		}
	}
}

// Garbling uses a different strategy
// Whereby each 512-bit register represents the AES operations for that gate

// width in number of tables
// bufferOffset in bytes
template<size_t width>
void FixedKeyLTGarblingVaesProcessor::computeOutKeysAndTable(uint32_t tableCounter, size_t numTablesInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer)
{
	const __m512i COUNTER_DIFF = _mm512_set_epi32(
		0, 0, 0, 2,
		0, 0, 0, 2,
		0, 0, 0, 2,
		0, 0, 0, 2
	);

	__m512i counter = _mm512_set_epi32(
		0, 0, 0, tableCounter * KEYS_PER_GATE_IN_TABLE + 1,
		0, 0, 0, tableCounter * KEYS_PER_GATE_IN_TABLE + 1,
		0, 0, 0, tableCounter * KEYS_PER_GATE_IN_TABLE,
		0, 0, 0, tableCounter * KEYS_PER_GATE_IN_TABLE);

	// note: this implementation heavily relies on the fact that
	// the optimizer notices that all w-indexed loop iterations are independent
	// *and* that it manages to assign different registers to each iteration

	__m512i data[width];
	__m512i keys[width];
	__m512i aes_keys[11];
	__m512i postMask[width];
	uint8_t* targetGateKey[width];
	uint8_t* targetGateKeyR[width];
	uint8_t rpbit[width];
	uint8_t finalMask[width];
	uint8_t* targetPiBit[width];

	// this broadcasts the key into the 512-bit registers
	for (size_t i = 0; i < 11; ++i)
	{
		__m128i temp_key = _mm_load_si128((__m128i*)(m_fixedKeyProvider.getExpandedStaticKey() + i * 16));
		aes_keys[i] = _mm512_broadcast_i32x4(temp_key);
	}

	const __m128i R = _mm_loadu_si128((__m128i*)m_globalRandomOffset);
	const __m512i wideR = _mm512_broadcast_i32x4(R);

	uint32_t currentOffset = simdStartOffset;
	uint32_t currentGateIdx = queueStartIndex;
	uint8_t* gtptr = tableBuffer + 16 * KEYS_PER_GATE_IN_TABLE * tableCounter;

	for (size_t i = 0; i < numTablesInBatch; i += width)
	{
		// TODO: optimize this to detect and exploit simd gates
		// saving vGate, parent base pointer and owning gate lookups
		for (size_t w = 0; w < width; ++w)
		{
			const GATE* currentGate = m_tableGateQueue[currentGateIdx];
			const uint32_t leftParentId = currentGate->ingates.inputs.twin.left;
			const uint32_t rightParentId = currentGate->ingates.inputs.twin.right;
			const GATE* leftParent = &m_vGates[leftParentId];
			const GATE* rightParent = &m_vGates[rightParentId];
			const uint8_t* leftParentKey = leftParent->gs.yinput.outKey[0] + 16 * currentOffset;
			const uint8_t* rightParentKey = rightParent->gs.yinput.outKey[0] + 16 * currentOffset;
			const uint8_t lpbit = leftParent->gs.yinput.pi[currentOffset];
			const uint8_t lpbit11 = (lpbit << 1) | lpbit;
			const uint8_t rpbitLocal = rightParent->gs.yinput.pi[currentOffset];
			const uint8_t rpbit11 = (rpbitLocal << 1) | rpbitLocal;
			rpbit[w] = rpbit11;

			currentGate->gs.yinput.pi[currentOffset] = lpbit & rpbitLocal;

			const __m128i lowerLow = _mm_loadu_si128((__m128i*)leftParentKey);
			const __m128i upperLow = _mm_xor_si128(lowerLow, R);
			const __m128i lowerUpper = _mm_loadu_si128((__m128i*)rightParentKey);
			const __m128i upperUpper = _mm_xor_si128(lowerUpper, R);

			const __m128i toBeInserted = _mm_mask_blend_epi64(lpbit11, lowerLow, upperLow);
			postMask[w] = _mm512_inserti32x4(wideR, toBeInserted, 2);

			/*if (lpbit)
				postMask[w] = _mm512_inserti32x4(wideR, upperLow, 2);
			else
				postMask[w] = _mm512_inserti32x4(wideR, lowerLow, 2);*/

			targetGateKey[w] = currentGate->gs.yinput.outKey[0] + 16 * currentOffset;
			targetGateKeyR[w] = currentGate->gs.yinput.outKey[1] + 16 * currentOffset;
			targetPiBit[w] = currentGate->gs.yinput.pi + currentOffset;

			const uint8_t lsbitANDrsbit = (leftParentKey[15] & 0x01) & (rightParentKey[15] & 0x01);
			const uint8_t lsbitANDrsbit11 = (lsbitANDrsbit << 1) | lsbitANDrsbit;

			// these are the control bits for the application of the "postMask"
			finalMask[w] = lsbitANDrsbit11 | (rpbit11 << 2) | (0x03 << 4) | (lsbitANDrsbit11 << 6);

			// this is a "tree" construction due to the high latency of the inserti instructions
			keys[w] = _mm512_castsi128_si512(lowerLow);
			__m256i upper = _mm256_castsi128_si256(lowerUpper);
			upper = _mm256_inserti128_si256(upper, upperUpper, 1);
			keys[w] = _mm512_inserti32x4(keys[w], upperLow, 1);
			keys[w] = _mm512_inserti64x4(keys[w], upper, 1);

			currentOffset++;

			if (currentOffset >= currentGate->nvals)
			{
				currentGateIdx++;
				currentOffset = 0;
			}
		}

		for (size_t w = 0; w < width; ++w)
		{
			data[w] = counter;
			counter = _mm512_add_epi32(counter, COUNTER_DIFF);
		}

		for (size_t w = 0; w < width; ++w)
		{
			// this is the actual AES input
			data[w] = vaes_mix_keys(keys[w], data[w]);

			keys[w] = data[w]; // keep as a backup for post-whitening
		}

		vaes_encrypt_fixed_keys(width, aes_keys, data);

		for (size_t w = 0; w < width; ++w)
		{
			data[w] = _mm512_xor_si512(data[w], keys[w]);
		}


		for (size_t w = 0; w < width; ++w)
		{
			// intent (shuffling into, in terms of 128-bit lanes):
			// 2
			// 0
			// 3
			// 0
			const __m512i shuffleKey = _mm512_set_epi64(
				1, 0,
				7, 6,
				1, 0,
				5, 4
			);
			// this creates a shuffled copy of the encrypted data
			// to allow us to do the inter-lane XORs
			__m512i shuffledCopy = _mm512_permutexvar_epi64(shuffleKey, data[w]);
			__m512i firstXor = _mm512_xor_si512(data[w], shuffledCopy);

			__m512i secondXor = _mm512_mask_xor_epi64(firstXor, finalMask[w], firstXor, postMask[w]);
			_mm_storeu_si128((__m128i*)gtptr, _mm512_extracti32x4_epi32(secondXor, 1));
			gtptr += 16;
			_mm_storeu_si128((__m128i*)gtptr, _mm512_extracti32x4_epi32(secondXor, 2));
			gtptr += 16;

			const __m128i nonRpbitOutKey = _mm512_extracti32x4_epi32(secondXor, 0);
			const __m128i rpbitRXor = _mm512_extracti32x4_epi32(firstXor, 2);
			const __m128i rpbitLXor = _mm512_extracti32x4_epi32(secondXor, 3);
			__m128i outKey = _mm_mask_xor_epi64(nonRpbitOutKey, rpbit[w], rpbitLXor, rpbitRXor);

			/*
			if (rpbit[w]) {
				__m128i rXor = _mm512_extracti32x4_epi32(firstXor, 2);
				outKey = _mm512_extracti32x4_epi32(secondXor, 3);
				outKey = _mm_xor_si128(outKey, rXor);
			}
			else {
				outKey = _mm512_extracti32x4_epi32(secondXor, 0);
			}
			*/
			//uint8_t rBit = _mm_extract_epi8(R, 15) & 0x01;
			//assert(rBit == 1);
			uint8_t outWireBit = _mm_extract_epi8(outKey, 15) & 0x01;
			*targetPiBit[w] ^= outWireBit;
			const uint8_t amplifiedOutWireBit = (outWireBit << 1) | outWireBit;
			outKey = _mm_mask_xor_epi64(outKey, amplifiedOutWireBit, outKey, R);
			/*if (outWireBit) {
				outKey = _mm_xor_si128(outKey, R);
				*targetPiBit[w] ^= rBit;
			}*/

			_mm_storeu_si128((__m128i*)targetGateKey[w], outKey);
			_mm_storeu_si128((__m128i*)targetGateKeyR[w], _mm_xor_si128(outKey, R));
		}
	}
}

// the following ones works analogously to the Fixedkey one with a slightly different AES input combiner

// width in number of tables
// bufferOffset in bytes
template<size_t width>
void InputKeyLTGarblingVaesProcessor::computeOutKeysAndTable(uint32_t tableCounter, size_t numTablesInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer)
{
	const __m512i COUNTER_DIFF = _mm512_set_epi32(
		0, 0, 0, 2,
		0, 0, 0, 2,
		0, 0, 0, 2,
		0, 0, 0, 2
	);

	__m512i counter = _mm512_set_epi32(
		0, 0, 0, tableCounter * KEYS_PER_GATE_IN_TABLE + 1,
		0, 0, 0, tableCounter * KEYS_PER_GATE_IN_TABLE + 1,
		0, 0, 0, tableCounter * KEYS_PER_GATE_IN_TABLE,
		0, 0, 0, tableCounter * KEYS_PER_GATE_IN_TABLE);

	// note: this implementation heavily relies on the fact that
	// the optimizer notices that all w-indexed loop iterations are independent
	// *and* that it manages to assign different registers to each iteration

	__m512i data[width];
	__m512i keys[width];
	__m512i postMask[width];
	uint8_t* targetGateKey[width];
	uint8_t* targetGateKeyR[width];
	uint8_t rpbit[width];
	uint8_t finalMask[width];
	uint8_t* targetPiBit[width];

	const __m128i R = _mm_loadu_si128((__m128i*)m_globalRandomOffset);
	const __m512i wideR = _mm512_broadcast_i32x4(R);

	uint32_t currentOffset = simdStartOffset;
	uint32_t currentGateIdx = queueStartIndex;
	uint8_t* gtptr = tableBuffer + 16 * KEYS_PER_GATE_IN_TABLE * tableCounter;

	for (size_t i = 0; i < numTablesInBatch; i += width)
	{
		// TODO: optimize this to detect and exploit simd gates
		// saving vGate, parent base pointer and owning gate lookups
		for (size_t w = 0; w < width; ++w)
		{
			const GATE* currentGate = m_tableGateQueue[currentGateIdx];
			const uint32_t leftParentId = currentGate->ingates.inputs.twin.left;
			const uint32_t rightParentId = currentGate->ingates.inputs.twin.right;
			const GATE* leftParent = &m_vGates[leftParentId];
			const GATE* rightParent = &m_vGates[rightParentId];
			const uint8_t* leftParentKey = leftParent->gs.yinput.outKey[0] + 16 * currentOffset;
			const uint8_t* rightParentKey = rightParent->gs.yinput.outKey[0] + 16 * currentOffset;
			const uint8_t lpbit = leftParent->gs.yinput.pi[currentOffset];
			const uint8_t lpbit11 = (lpbit << 1) | lpbit;
			const uint8_t rpbitLocal = rightParent->gs.yinput.pi[currentOffset];
			const uint8_t rpbit11 = (rpbitLocal << 1) | rpbitLocal;
			rpbit[w] = rpbit11;

			currentGate->gs.yinput.pi[currentOffset] = lpbit & rpbitLocal;

			const __m128i lowerLow = _mm_loadu_si128((__m128i*)leftParentKey);
			const __m128i upperLow = _mm_xor_si128(lowerLow, R);
			const __m128i lowerUpper = _mm_loadu_si128((__m128i*)rightParentKey);
			const __m128i upperUpper = _mm_xor_si128(lowerUpper, R);

			const __m128i toBeInserted = _mm_mask_blend_epi64(lpbit11, lowerLow, upperLow);
			postMask[w] = _mm512_inserti32x4(wideR, toBeInserted, 2);

			/*if (lpbit)
				postMask[w] = _mm512_inserti32x4(wideR, upperLow, 2);
			else
				postMask[w] = _mm512_inserti32x4(wideR, lowerLow, 2);*/

			targetGateKey[w] = currentGate->gs.yinput.outKey[0] + 16 * currentOffset;
			targetGateKeyR[w] = currentGate->gs.yinput.outKey[1] + 16 * currentOffset;
			targetPiBit[w] = currentGate->gs.yinput.pi + currentOffset;

			const uint8_t lsbitANDrsbit = (leftParentKey[15] & 0x01) & (rightParentKey[15] & 0x01);
			const uint8_t lsbitANDrsbit11 = (lsbitANDrsbit << 1) | lsbitANDrsbit;

			finalMask[w] = lsbitANDrsbit11 | (rpbit11 << 2) | (0x03 << 4) | (lsbitANDrsbit11 << 6);

			// this is a "tree" construction due to the high latency of the inserti instructions
			keys[w] = _mm512_castsi128_si512(lowerLow);
			__m256i upper = _mm256_castsi128_si256(lowerUpper);
			upper = _mm256_inserti128_si256(upper, upperUpper, 1);
			keys[w] = _mm512_inserti32x4(keys[w], upperLow, 1);
			keys[w] = _mm512_inserti64x4(keys[w], upper, 1);

			currentOffset++;

			if (currentOffset >= currentGate->nvals)
			{
				currentGateIdx++;
				currentOffset = 0;
			}
		}

		for (size_t w = 0; w < width; ++w)
		{
			data[w] = counter;
			counter = _mm512_add_epi32(counter, COUNTER_DIFF);
		}

		vaes_encrypt_variable_keys(width, keys, data);

		for (size_t w = 0; w < width; ++w)
		{
			// intent (shuffling into, in terms of 128-bit lanes):
			// 2
			// 0
			// 3
			// 0
			const __m512i shuffleKey = _mm512_set_epi64(
				1, 0,
				7, 6,
				1, 0,
				5, 4
			);
			__m512i shuffledCopy = _mm512_permutexvar_epi64(shuffleKey, data[w]);
			__m512i firstXor = _mm512_xor_si512(data[w], shuffledCopy);

			__m512i secondXor = _mm512_mask_xor_epi64(firstXor, finalMask[w], firstXor, postMask[w]);
			_mm_storeu_si128((__m128i*)gtptr, _mm512_extracti32x4_epi32(secondXor, 1));
			gtptr += 16;
			_mm_storeu_si128((__m128i*)gtptr, _mm512_extracti32x4_epi32(secondXor, 2));
			gtptr += 16;
			const __m128i nonRpbitOutKey = _mm512_extracti32x4_epi32(secondXor, 0);
			const __m128i rpbitRXor = _mm512_extracti32x4_epi32(firstXor, 2);
			const __m128i rpbitLXor = _mm512_extracti32x4_epi32(secondXor, 3);
			__m128i outKey = _mm_mask_xor_epi64(nonRpbitOutKey, rpbit[w], rpbitLXor, rpbitRXor);
			uint8_t outWireBit = _mm_extract_epi8(outKey, 15) & 0x01;
			*targetPiBit[w] ^= outWireBit;
			const uint8_t amplifiedOutWireBit = (outWireBit << 1) | outWireBit;
			outKey = _mm_mask_xor_epi64(outKey, amplifiedOutWireBit, outKey, R);

			_mm_storeu_si128((__m128i*)targetGateKey[w], outKey);
			_mm_storeu_si128((__m128i*)targetGateKeyR[w], _mm_xor_si128(outKey, R));
		}
	}
}

template<size_t width>
inline void MILTEvaluatingVaesProcessor::computeAESOutKeys(uint32_t tableCounter, size_t queueStartIndex, size_t simdStartOffset, size_t numTablesInBatch, const uint8_t* receivedTables)
{
	constexpr size_t div_width = (width + 3) / 4; // ceiling division
	constexpr size_t num_buffer_words = std::min(width, size_t(4));

	static_assert((width < 4) || (width % 4 == 0), "This implementation only supports multiplies of 4 or values smaller than 4.");

	const __m512i ONE = _mm512_set_epi32(
		0, 0, 0, 1,
		0, 0, 0, 1,
		0, 0, 0, 1,
		0, 0, 0, 1);

	// the offset calculation WILL break down if the above static assert is violated
	// Example: width==7
	// then the first 4 would need an offset of 8
	// and then the second 3 would need an offset of 6 so the next round has the right offset
	// we *could* fix this with a div_width array of offsets where the first div_width-1 elements have the 8
	// and the remainder has (width%4)*2 however we don't need such weird instances
	constexpr size_t offset = std::min(size_t(4) * KEYS_PER_GATE_IN_TABLE, width * KEYS_PER_GATE_IN_TABLE);

	const __m512i FULL_OFFSET = _mm512_set_epi32(
		0, 0, 0, offset,
		0, 0, 0, offset,
		0, 0, 0, offset,
		0, 0, 0, offset);



	__m512i counter = _mm512_set_epi32(
		0, 0, 0, (tableCounter + 3) * KEYS_PER_GATE_IN_TABLE,
		0, 0, 0, (tableCounter + 2) * KEYS_PER_GATE_IN_TABLE,
		0, 0, 0, (tableCounter + 1) * KEYS_PER_GATE_IN_TABLE,
		0, 0, 0, (tableCounter + 0) * KEYS_PER_GATE_IN_TABLE);

	const __m128i idxOffset_small = _mm_loadu_si128((__m128i*)m_globalIndexOffset);
	const __m512i idxOffset = _mm512_broadcast_i32x4(idxOffset_small);
	counter = _mm512_add_epi32(counter, idxOffset);

	// note: this implementation heavily relies on the fact that
	// the optimizer notices that all w-indexed loop iterations are independent
	// *and* that it manages to assign different registers to each iteration

	__m512i leftData[div_width];
	__m512i rightData[div_width];
	__m512i leftKeys[div_width];
	__m512i rightKeys[div_width];
	__m512i finalMask[div_width];
	__m512i leftWhitening[div_width];
	__m512i rightWhitening[div_width];
	uint8_t* targetGateKey[width];
	const uint8_t* gtptr = receivedTables + tableCounter * KEYS_PER_GATE_IN_TABLE * 16;

	size_t currentGateIdx = queueStartIndex;
	uint32_t currentOffset = simdStartOffset;

	for (size_t i = 0; i < numTablesInBatch; i += width)
	{
		// TODO: optimize using bigger vector loads potentially?
		for (size_t w = 0; w < div_width; ++w)
		{
			for (size_t k = 0; k < num_buffer_words; ++k) {
				const GATE* currentGate = m_gateQueue[currentGateIdx];
				const uint32_t leftParentId = currentGate->ingates.inputs.twin.left;
				const uint32_t rightParentId = currentGate->ingates.inputs.twin.right;
				const GATE* leftParent = &m_vGates[leftParentId];
				const GATE* rightParent = &m_vGates[rightParentId];
				const uint8_t* leftParentKey = leftParent->gs.yval + 16 * currentOffset;
				const uint8_t* rightParentKey = rightParent->gs.yval + 16 * currentOffset;

				const __m128i leftParentKeyLocal = _mm_loadu_si128((__m128i*)leftParentKey);
				leftKeys[w] = mm512_insert_128(leftKeys[w], leftParentKeyLocal, k);
				const __m128i rightParentKeyLocal = _mm_loadu_si128((__m128i*)rightParentKey);
				rightKeys[w] = mm512_insert_128(rightKeys[w], rightParentKeyLocal, k);

				targetGateKey[4 * w + k] = currentGate->gs.yval + 16 * currentOffset;

				const uint8_t lpbit = leftParentKey[15] & 0x01;
				const uint8_t lpbit11 = (lpbit << 1) | lpbit;
				const uint8_t rpbit = rightParentKey[15] & 0x01;
				const uint8_t rpbit11 = (rpbit << 1) | rpbit;

				__m128i finalMaskLocal = _mm_maskz_loadu_epi64(lpbit11, (__m128i*)gtptr);
				gtptr += 16;
				const __m128i rightTable = _mm_loadu_si128((__m128i*)gtptr);
				const __m128i rightMaskUpdate = _mm_xor_si128(rightTable, leftParentKeyLocal);
				finalMaskLocal = _mm_mask_xor_epi64(finalMaskLocal, rpbit11, finalMaskLocal, rightMaskUpdate);
				gtptr += 16;

				finalMask[w] = mm512_insert_128(finalMask[w], finalMaskLocal, k);

				currentOffset++;

				if (currentOffset >= currentGate->nvals)
				{
					currentGateIdx++;
					currentOffset = 0;
				}
			}
		}

		for (size_t w = 0; w < div_width; ++w)
		{
			// use this because addition has a latency of 1 and a throughput of 0.5 CPI
			leftData[w] = counter;
			rightData[w] = _mm512_add_epi32(counter, ONE);
			counter = _mm512_add_epi32(counter, FULL_OFFSET);

			leftKeys[w] = vaes_orthomorphism(leftKeys[w]);
			rightKeys[w] = vaes_orthomorphism(rightKeys[w]);
			leftWhitening[w] = leftKeys[w];
			rightWhitening[w] = rightKeys[w];
		}

		vaes_encrypt_variable_keys(div_width, leftData, leftKeys, rightData, rightKeys);

		for (size_t w = 0; w < div_width; ++w)
		{
			leftKeys[w] = _mm512_xor_si512(leftKeys[w], leftWhitening[w]);
			rightKeys[w] = _mm512_xor_si512(rightKeys[w], rightWhitening[w]);

			leftKeys[w] = _mm512_xor_si512(leftKeys[w], rightKeys[w]);
			leftKeys[w] = _mm512_xor_si512(leftKeys[w], finalMask[w]);
		}

		for (size_t w = 0; w < div_width; ++w)
		{
			for (size_t k = 0; k < num_buffer_words; ++k) {
				const __m128i extracted = mm512_extract_128(leftKeys[w], k);
				_mm_storeu_si128((__m128i*)(targetGateKey[4 * w + k]), extracted);
			}
		}
	}
}

template<size_t width>
void MILTGarblingVaesProcessor::computeOutKeysAndTable(uint32_t tableCounter, size_t numTablesInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer)
{
	static_assert(width < 2 || width % 2 == 0, "Only values smaller than 2 or multiples thereof are supported");

	constexpr size_t div_width = (width + 1) / 2; // ceiling division
	constexpr size_t num_buffer_words = std::min(width, size_t(2));
	constexpr size_t nbw = num_buffer_words;
	constexpr size_t offset = num_buffer_words * 2;

	const __m512i COUNTER_DIFF = _mm512_set_epi32(
		0, 0, 0, offset,
		0, 0, 0, offset,
		0, 0, 0, offset,
		0, 0, 0, offset
	);

	__m512i counter = _mm512_set_epi32(
		0, 0, 0, tableCounter * KEYS_PER_GATE_IN_TABLE + 3,
		0, 0, 0, tableCounter * KEYS_PER_GATE_IN_TABLE + 2,
		0, 0, 0, tableCounter * KEYS_PER_GATE_IN_TABLE + 1,
		0, 0, 0, tableCounter * KEYS_PER_GATE_IN_TABLE + 0);

	const __m128i idxOffset_small = _mm_loadu_si128((__m128i*)m_globalIndexOffset);
	const __m512i idxOffset = _mm512_broadcast_i32x4(idxOffset_small);
	counter = _mm512_add_epi32(counter, idxOffset);

	// note: this implementation heavily relies on the fact that
	// the optimizer notices that all w-indexed loop iterations are independent
	// *and* that it manages to assign different registers to each iteration

	__m512i data[div_width];
	__m512i keys[2 * div_width];
	__m512i whitening[2 * div_width];
	__m512i tableMask[div_width];
	uint8_t* targetGateKey[width];
	uint8_t* targetGateKeyR[width];
	uint8_t rpbit[width];
	uint8_t lsbitANDrsbit[width];
	uint8_t* targetPiBit[width];

	const __m128i R = _mm_loadu_si128((__m128i*)m_globalRandomOffset);
	const __m512i wideR = _mm512_broadcast_i32x4(R);

	uint32_t currentOffset = simdStartOffset;
	uint32_t currentGateIdx = queueStartIndex;
	uint8_t* gtptr = tableBuffer + 16 * KEYS_PER_GATE_IN_TABLE * tableCounter;

	for (size_t i = 0; i < numTablesInBatch; i += width)
	{
		// TODO: optimize this to detect and exploit simd gates
		// saving vGate, parent base pointer and owning gate lookups
		for (size_t w = 0; w < div_width; ++w)
		{
			for (size_t k = 0; k < num_buffer_words; ++k) {
				const GATE* currentGate = m_tableGateQueue[currentGateIdx];
				const uint32_t leftParentId = currentGate->ingates.inputs.twin.left;
				const uint32_t rightParentId = currentGate->ingates.inputs.twin.right;
				const GATE* leftParent = &m_vGates[leftParentId];
				const GATE* rightParent = &m_vGates[rightParentId];
				const uint8_t* leftParentKey = leftParent->gs.yinput.outKey[0] + 16 * currentOffset;
				const uint8_t* rightParentKey = rightParent->gs.yinput.outKey[0] + 16 * currentOffset;
				const uint8_t lpbit = leftParent->gs.yinput.pi[currentOffset];
				const uint8_t lpbit11 = (lpbit << 1) | lpbit;
				const uint8_t rpbitLocal = rightParent->gs.yinput.pi[currentOffset];
				const uint8_t rpbit11 = (rpbitLocal << 1) | rpbitLocal;
				rpbit[w * nbw + k] = rpbit11;

				currentGate->gs.yinput.pi[currentOffset] = lpbit & rpbitLocal;

				const __m128i lowerLow = _mm_loadu_si128((__m128i*)leftParentKey);
				const __m128i upperLow = _mm_xor_si128(lowerLow, R);
				const __m128i lowerUpper = _mm_loadu_si128((__m128i*)rightParentKey);
				const __m128i upperUpper = _mm_xor_si128(lowerUpper, R);

				const __m128i toBeInsertedUpper = _mm_mask_blend_epi64(lpbit11, lowerLow, upperLow);
				tableMask[w] = mm512_insert_128(tableMask[w], toBeInsertedUpper, 2 * k + 1);
				const __m128i toBeInsertedLower = _mm_mask_blend_epi64(rpbit11, _mm_setzero_si128(), R);
				tableMask[w] = mm512_insert_128(tableMask[w], toBeInsertedLower, 2 * k + 0);

				targetGateKey[w * nbw + k] = currentGate->gs.yinput.outKey[0] + 16 * currentOffset;
				targetGateKeyR[w * nbw + k] = currentGate->gs.yinput.outKey[1] + 16 * currentOffset;
				targetPiBit[w * nbw + k] = currentGate->gs.yinput.pi + currentOffset;

				const uint8_t lsbitANDrsbitLocal = (leftParentKey[15] & 0x01) & (rightParentKey[15] & 0x01);
				const uint8_t lsbitANDrsbit11 = (lsbitANDrsbitLocal << 1) | lsbitANDrsbitLocal;

				lsbitANDrsbit[w * nbw + k] = lsbitANDrsbit11;

				keys[2 * w + 0] = mm512_insert_128(keys[2 * w + 0], lowerLow, 2 * k + 0);
				keys[2 * w + 0] = mm512_insert_128(keys[2 * w + 0], lowerUpper, 2 * k + 1);
				keys[2 * w + 1] = mm512_insert_128(keys[2 * w + 1], upperLow, 2 * k + 0);
				keys[2 * w + 1] = mm512_insert_128(keys[2 * w + 1], upperUpper, 2 * k + 1);

				currentOffset++;

				if (currentOffset >= currentGate->nvals)
				{
					currentGateIdx++;
					currentOffset = 0;
				}
			}
		}

		for (size_t w = 0; w < div_width; ++w)
		{
			data[w] = counter;
			counter = _mm512_add_epi32(counter, COUNTER_DIFF);
		}

		for (size_t w = 0; w < 2 * div_width; ++w)
		{
			keys[w] = vaes_orthomorphism(keys[w]);
			whitening[w] = keys[w];
		}

		vaes_encrypt_variable_keys_variable_data<div_width, 2>(data, keys);

		for (size_t w = 0; w < div_width; ++w)
		{
			keys[2 * w + 0] = _mm512_xor_si512(keys[2 * w + 0], whitening[2 * w + 0]);
			keys[2 * w + 1] = _mm512_xor_si512(keys[2 * w + 1], whitening[2 * w + 1]);

			const __m512i pretable = _mm512_xor_si512(keys[2 * w + 0], keys[2 * w + 1]);
			const __m512i tables = _mm512_xor_si512(pretable, tableMask[w]);

			if constexpr (width == 1) {
				_mm256_storeu_si256((__m256i*)gtptr, _mm512_extracti64x4_epi64(tables, 0));
				gtptr += 32;
			}
			else {
				_mm512_storeu_si512((__m512i*)gtptr, tables);
				gtptr += 64;
			}

			for (size_t k = 0; k < num_buffer_words; ++k) {
				const __m128i data0 = mm512_extract_128(keys[2 * w + 0], 2 * k);
				const __m128i data3 = mm512_extract_128(keys[2 * w + 1], 2 * k + 1);
				const __m128i data2 = mm512_extract_128(keys[2 * w + 0], 2 * k + 1);
				const __m128i rightData = _mm_mask_blend_epi64(rpbit[nbw * w + k], data2, data3);
				__m128i outKey = _mm_xor_si128(data0, rightData);
				outKey = _mm_mask_xor_epi64(outKey, lsbitANDrsbit[w * nbw + k], outKey, R);
				const __m128i rXor = mm512_extract_128(pretable, 2 * k + 1);
				outKey = _mm_mask_xor_epi64(outKey, rpbit[nbw * w + k], outKey, rXor);


				uint8_t outWireBit = _mm_extract_epi8(outKey, 15) & 0x01;
				*targetPiBit[w*nbw+k] ^= outWireBit;
				const uint8_t amplifiedOutWireBit = (outWireBit << 1) | outWireBit;
				outKey = _mm_mask_xor_epi64(outKey, amplifiedOutWireBit, outKey, R);

				_mm_storeu_si128((__m128i*)targetGateKey[w * nbw + k], outKey);
				_mm_storeu_si128((__m128i*)targetGateKeyR[w * nbw + k], _mm_xor_si128(outKey, R));
			}
		}
	}
}
