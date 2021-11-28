#ifndef __VAES_HELPERS_H__
#define __VAES_HELPERS_H__

#include <immintrin.h>
#include <iostream>
#include <iomanip>

// this is a trick from Rust:
// assumes that this function will be inlined *and*
// that the loop variable which is fed into location will be unrolled
static inline __attribute__((always_inline)) __m512i mm512_insert_128(__m512i baseline, __m128i word, size_t location) {
	switch (location & 0x03)
	{
	case 0:
		return _mm512_inserti32x4(baseline, word, 0);
	case 1:
		return _mm512_inserti32x4(baseline, word, 1);
	case 2:
		return _mm512_inserti32x4(baseline, word, 2);
	case 3:
		return _mm512_inserti32x4(baseline, word, 3);
	}
}

// this is a trick from Rust:
// assumes that this function will be inlined *and*
// that the loop variable which is fed into location will be unrolled
static inline __attribute__((always_inline)) __m128i mm512_extract_128(__m512i baseline, size_t location) {
	switch (location & 0x03)
	{
	case 0:
		return _mm512_extracti32x4_epi32(baseline, 0);
	case 1:
		return _mm512_extracti32x4_epi32(baseline, 1);
	case 2:
		return _mm512_extracti32x4_epi32(baseline, 2);
	case 3:
		return _mm512_extracti32x4_epi32(baseline, 3);
	}
}

static inline void PrintKey(__m512i data) {
	uint8_t key[64];
	_mm512_storeu_si512((__m512i*)key, data);

	for (int j = 0; j < 64; j += 16)
	{
		for (uint32_t i = 0; i < 16; i++) {
			std::cout << std::setw(2) << std::setfill('0') << (std::hex) << (uint32_t)key[i + j];
		}
		std::cout << std::endl;
	}

	std::cout << (std::dec);
}

// this is a generic XOR operation
// that can be invoked as vaes_xor(w,left1,right1,left2,right2)
// and then produces right1[w]=left1[w]^right1[w] and right2[w]=left2[w]^right2[w]
template<typename... ops>
static inline __attribute__((always_inline)) void vaes_xor(
	const size_t w,
	__m512i key[],
	__m512i data[],
	ops... remainder
) {
	data[w] = _mm512_xor_si512(key[w], data[w]);
	if constexpr (sizeof...(ops) > 0) {
		vaes_xor(w, remainder...);
	}
}

// this performs one round of the fast key expansion and on-the-fly encryption by Gueron
// https://eprint.iacr.org/2015/751
// with 512-bit registers and not updating rcon
// the invocation uses the first five arguments as constant
// and can then process arbitrary key-data register array pairs
// the first argument controls whether to use aesenc or aesenclast
template< typename enc_op_t, typename... ops>
static inline __attribute__((always_inline)) void vaes_encrypt_one_round_variable_keys(
	const enc_op_t enc_op,
	const __m512i shuffle_mask,
	const __m512i con3,
	const __m512i rcon,
	const size_t w,
	__m512i key[],
	__m512i data[],
	ops... remainder) {
	static_assert(sizeof...(ops) % 2 == 0);

	__m512i temp2, temp3;
	temp2 = _mm512_shuffle_epi8(key[w], shuffle_mask);
	temp2 = _mm512_aesenclast_epi128(temp2, rcon);
	// the rcon update used to be here, moved it out because otherwise correctness would fail due to the inner loop
	__m512i globAux = _mm512_slli_epi64(key[w], 32);
	key[w] = _mm512_xor_si512(globAux, key[w]);
	globAux = _mm512_shuffle_epi8(key[w], con3);
	key[w] = _mm512_xor_si512(globAux, key[w]);
	key[w] = _mm512_xor_si512(temp2, key[w]);

	data[w] = enc_op(data[w], key[w]);

	if constexpr (sizeof...(ops) > 0) {
		vaes_encrypt_one_round_variable_keys(enc_op, shuffle_mask, con3, rcon, w, remainder...);
	}
}

// this does a full interleaved variable key encryption using multiple variable arrays
// i.e. you can invoke as vaes_encrypt_variable_keys(width,data1,key1,data2,key2) and it will encrypt data1 with key1 and data2 with key2
// widths in 512 bit words
template<typename ... ops>
static inline __attribute__((always_inline)) void vaes_encrypt_variable_keys(const size_t width, ops... keydata) {
	// this uses the fast AES key expansion (i.e. not using keygenassist) from
	// https://www.intel.com/content/dam/doc/white-paper/advanced-encryption-standard-new-instructions-set-paper.pdf
	// page 37

	for (size_t w = 0; w < width; ++w) {
		vaes_xor(w, keydata...);
	}

	__m512i rcon = _mm512_set1_epi32(1);
	const __m512i shuffle_mask = _mm512_set1_epi32(0x0c0f0e0d);
	const __m128i con3 = _mm_set_epi32(0x07060504, 0x07060504, 0x0ffffffff, 0x0ffffffff);
	const __m512i wideCon3 = _mm512_broadcast_i32x4(con3);
	const __m512i rcon_multiplier = _mm512_set1_epi8(2);

	for (size_t r = 1; r < 10; ++r)
	{
		for (size_t w = 0; w < width; ++w)
		{
			const auto enc_op_lambda = [](const __m512i data, const __m512i key) {return _mm512_aesenc_epi128(data, key); };
			vaes_encrypt_one_round_variable_keys(enc_op_lambda, shuffle_mask, wideCon3, rcon, w, keydata...);
		}

		rcon = _mm512_gf2p8mul_epi8(rcon, rcon_multiplier);
	}

	for (size_t w = 0; w < width; ++w)
	{
		const auto enc_op_lambda = [](const __m512i data, const __m512i key) {return _mm512_aesenclast_epi128(data, key); };
		vaes_encrypt_one_round_variable_keys(enc_op_lambda, shuffle_mask, wideCon3, rcon, w, keydata...);
	}
}

// this is more simple than the above
// but can encrypt multiple array entries with the same freshly expanded key using vectorization
template<size_t key_width, size_t data_per_key>
static inline __attribute__((always_inline)) void vaes_encrypt_variable_keys_variable_data(__m512i keys[key_width], __m512i data[key_width * data_per_key]) {
	// this uses the fast AES keys expansion (i.e. not using keygenassist) from
	// https://www.intel.com/content/dam/doc/white-paper/advanced-encryption-standard-new-instructions-set-paper.pdf
	// page 37

	for (size_t w = 0; w < key_width; ++w) {
		for (size_t d = 0; d < data_per_key; ++d)
			data[w * data_per_key + d] = _mm512_xor_si512(data[w * data_per_key + d], keys[w]);
	}

	__m512i rcon = _mm512_set1_epi32(1);
	const __m512i shuffle_mask = _mm512_set1_epi32(0x0c0f0e0d);
	const __m128i smallCon3 = _mm_set_epi32(0x07060504, 0x07060504, 0x0ffffffff, 0x0ffffffff);
	const __m512i con3 = _mm512_broadcast_i32x4(smallCon3);
	const __m512i rcon_multiplier = _mm512_set1_epi8(2);

	for (size_t r = 1; r < 10; ++r)
	{
		for (size_t w = 0; w < key_width; ++w)
		{
			__m512i temp2, temp3;
			temp2 = _mm512_shuffle_epi8(keys[w], shuffle_mask);
			temp2 = _mm512_aesenclast_epi128(temp2, rcon);
			// the rcon update used to be here, moved it out because otherwise correctness would fail due to the inner loop
			__m512i globAux = _mm512_slli_epi64(keys[w], 32);
			keys[w] = _mm512_xor_si512(globAux, keys[w]);
			globAux = _mm512_shuffle_epi8(keys[w], con3);
			keys[w] = _mm512_xor_si512(globAux, keys[w]);
			keys[w] = _mm512_xor_si512(temp2, keys[w]);

			for (size_t d = 0; d < data_per_key; ++d)
				data[w * data_per_key + d] = _mm512_aesenc_epi128(data[w * data_per_key + d], keys[w]);
		}

		rcon = _mm512_gf2p8mul_epi8(rcon, rcon_multiplier);
	}

	for (size_t w = 0; w < key_width; ++w)
	{
		__m512i temp2, temp3;
		temp2 = _mm512_shuffle_epi8(keys[w], shuffle_mask);
		temp2 = _mm512_aesenclast_epi128(temp2, rcon);
		// the rcon update used to be here, moved it out because otherwise correctness would fail due to the inner loop
		__m512i globAux = _mm512_slli_epi64(keys[w], 32);
		keys[w] = _mm512_xor_si512(globAux, keys[w]);
		globAux = _mm512_shuffle_epi8(keys[w], con3);
		keys[w] = _mm512_xor_si512(globAux, keys[w]);
		keys[w] = _mm512_xor_si512(temp2, keys[w]);

		for (size_t d = 0; d < data_per_key; ++d)
			data[w * data_per_key + d] = _mm512_aesenclast_epi128(data[w * data_per_key + d], keys[w]);
	}
}

// this is a helper for variable variable-register fixed key encryption
template<typename map_t, typename ... ops>
static inline __attribute__((always_inline)) void apply_op(const map_t map, const size_t w, __m512i value[], ops... remainder) {
	value[w] = map(value[w]);
	if constexpr (sizeof...(ops) > 0) {
		apply_op(map, w, remainder...);
	}
}

// fixed key encryption with a variable amount of arguments
// i.e. vaes_encrypt_fixed_keys(width,round_keys,data1,data2,data3)
// widths in 512 bit words
template<typename ... ops>
static inline __attribute__((always_inline)) void vaes_encrypt_fixed_keys(const size_t width, const __m512i round_keys[11], ops... data) {
	for (size_t w = 0; w < width; ++w)
		apply_op([round_keys](const __m512i data) {return _mm512_xor_si512(data, round_keys[0]); }, w, data...);
	for(size_t r=0;r<10;++r)
		for (size_t w = 0; w < width; ++w)
			apply_op([round_keys,r](const __m512i data) {return _mm512_aesenc_epi128(data, round_keys[r]); }, w, data...);
	for (size_t w = 0; w < width; ++w)
		apply_op([round_keys](const __m512i data) {return _mm512_aesenclast_epi128(data, round_keys[10]); }, w, data...);
}

#endif