#ifndef EMP_FERRET_TWO_KEY_PRP_H__
#define EMP_FERRET_TWO_KEY_PRP_H__

#include "emp-tool/emp-tool.h"
#include <immintrin.h>
using namespace emp;

//kappa->2kappa PRG, implemented as G(k) = PRF_seed0(k)\xor k || PRF_seed1(k)\xor k
class TwoKeyPRP { public:
	// this holds the pre-distributed VAES round keys
	__m512i vaes_key[11];
	AES_KEY aes_key[2];

	__attribute__((target("avx512f")))
	TwoKeyPRP(block seed0, block seed1) {
		AES_set_encrypt_key((const block)seed0, aes_key);
		AES_set_encrypt_key((const block)seed1, &aes_key[1]);
		for (size_t r = 0; r < 11; ++r) {
			__m512i wide_left = _mm512_broadcast_i32x4(aes_key[0].rd_key[r]);
			__m512i wide_right = _mm512_broadcast_i32x4(aes_key[1].rd_key[r]);
			// this yields left, right, left, right in the lanes for the round keys
			__m512i round_key = _mm512_mask_blend_epi64(0xCC, wide_left, wide_right);
			_mm512_storeu_si512(&vaes_key[r], round_key);
		}
	}

	// the following two replace the below explicit 1to2 and 2to4 variants using loop-based processing
	// This VAES version is slightly preferred for larger values of width
	template<size_t width> __attribute__((target("avx512f")))
	void node_expand_double_vaes(block* children, block* parent) {
		__m512i tmp[width];
		__m512i whitening[width];
		for (size_t w = 0; w < width; ++w) {
			__m512i wide_left = _mm512_broadcast_i32x4(parent[2*w+0]);
			__m512i wide_right = _mm512_broadcast_i32x4(parent[2 * w + 1]);
			// this yields left, left, right, right for the input blocks
			tmp[w] = _mm512_mask_blend_epi64(0xF0, wide_left, wide_right);
			whitening[w] = tmp[w];
		}
		// paired with the initialization this gives us each parent with each seed key being encrypted
		permute_blocks_vaes<width>(tmp);
		for (size_t w = 0; w < width; ++w) {
			__m512i writeOut = whitening[w] ^ tmp[w];
			_mm512_storeu_si512((__m512i*)&children[4 * w], writeOut);
		}
	}

	template<size_t width>
	void node_expand_double(block* children, block* parent) {
		block tmp[2*width];
		for (int w = width-1; w >= 0; --w) {
			tmp[2 * w + 1] = children[2 * w + 1] = parent[w];
			tmp[2 * w + 0] = children[2 * w + 0] = parent[w];
		}
		permute_blocks<2*width>(tmp);
		for (int w = 2*width-1; w >= 0; --w) {
			children[w] = children[w] ^ tmp[w];
		}
	}

	void node_expand_1to2(block *children, block parent) {
		block tmp[2];
		tmp[0] = children[0] = parent;
		tmp[1] = children[1] = parent;
		permute_block_2blks(tmp);
		children[0] = children[0] ^ tmp[0];
		children[1] = children[1] ^ tmp[1];
	}

	void node_expand_2to4(block *children, block *parent) {
		block tmp[4];
		tmp[3] = children[3] = parent[1];
		tmp[2] = children[2] = parent[1];
		tmp[1] = children[1] = parent[0];
		tmp[0] = children[0] = parent[0];
		permute_block_4blks(tmp);
		children[3] = children[3] ^ tmp[3];
		children[2] = children[2] ^ tmp[2];
		children[1] = children[1] ^ tmp[1];
		children[0] = children[0] ^ tmp[0];
	}

#ifdef __x86_64__
	template <size_t width> __attribute__((target("aes,sse2")))
		inline void permute_blocks(block* blks) {
		static_assert(width % 2 == 0, "only multiplies of 2 as width are supported");
		for (size_t w = 0; w < width; ++w) {
			blks[w] = _mm_xor_si128(blks[w], aes_key[w % 2].rd_key[0]);
		}
		for (size_t r = 1; r < 10; ++r) {
			for (size_t w = 0; w < width; ++w) {
				blks[w] = _mm_aesenc_si128(blks[w], aes_key[w%2].rd_key[r]);
			}
		}
		for (size_t w = 0; w < width; ++w) {
			blks[w] = _mm_aesenclast_si128(blks[w], aes_key[w % 2].rd_key[10]);
		}
	}

	template <size_t width> __attribute__((target("avx512f,vaes")))
		inline void permute_blocks_vaes(__m512i blks[width]) {
		__m512i round_key = _mm512_loadu_si512(&vaes_key[0]);
		for (size_t w = 0; w < width; ++w) {
			blks[w] = _mm512_xor_si512(blks[w], round_key);
		}
		for (size_t r = 1; r < 10; ++r) {
			round_key = _mm512_loadu_si512(&vaes_key[r]);
			for (size_t w = 0; w < width; ++w) {
				blks[w] = _mm512_aesenc_epi128(blks[w], round_key);
			}
		}
		round_key = _mm512_loadu_si512(&vaes_key[10]);
		for (size_t w = 0; w < width; ++w) {
			blks[w] = _mm512_aesenclast_epi128(blks[w], round_key);
		}
	}

	__attribute__((target("aes,sse2")))
	inline void permute_block_4blks(block *blks) {
		blks[0] = _mm_xor_si128(blks[0], aes_key[0].rd_key[0]);
		blks[1] = _mm_xor_si128(blks[1], aes_key[1].rd_key[0]);
		blks[2] = _mm_xor_si128(blks[2], aes_key[0].rd_key[0]);
		blks[3] = _mm_xor_si128(blks[3], aes_key[1].rd_key[0]);
		for (unsigned int j = 1; j < aes_key[0].rounds; ++j) {
			blks[0] = _mm_aesenc_si128(blks[0], aes_key[0].rd_key[j]);
			blks[1] = _mm_aesenc_si128(blks[1], aes_key[1].rd_key[j]);
			blks[2] = _mm_aesenc_si128(blks[2], aes_key[0].rd_key[j]);
			blks[3] = _mm_aesenc_si128(blks[3], aes_key[1].rd_key[j]);
		}
		blks[0] = _mm_aesenclast_si128(blks[0], aes_key[0].rd_key[aes_key[0].rounds]);
		blks[1] = _mm_aesenclast_si128(blks[1], aes_key[1].rd_key[aes_key[1].rounds]);
		blks[2] = _mm_aesenclast_si128(blks[2], aes_key[0].rd_key[aes_key[0].rounds]);
		blks[3] = _mm_aesenclast_si128(blks[3], aes_key[1].rd_key[aes_key[1].rounds]);
	}

	__attribute__((target("aes,sse2")))
	inline void permute_block_2blks(block *blks) {
		blks[0] = _mm_xor_si128(blks[0], aes_key[0].rd_key[0]);
		blks[1] = _mm_xor_si128(blks[1], aes_key[1].rd_key[0]);
		for (unsigned int j = 1; j < aes_key[0].rounds; ++j) {
			blks[0] = _mm_aesenc_si128(blks[0], aes_key[0].rd_key[j]);
			blks[1] = _mm_aesenc_si128(blks[1], aes_key[1].rd_key[j]);
		}
		blks[0] = _mm_aesenclast_si128(blks[0], aes_key[0].rd_key[aes_key[0].rounds]);
		blks[1] = _mm_aesenclast_si128(blks[1], aes_key[1].rd_key[aes_key[1].rounds]);
	}
#elif __aarch64__
	inline void permute_block_4blks(block *_blks) {
		uint8x16_t * blks = (uint8x16_t*)(_blks);
		for (unsigned int i = 0; i < 10; ++i) {
			blks[0] = vaesmcq_u8(vaeseq_u8(blks[0], vreinterpretq_u8_m128i(aes_key[0].rd_key[i])));
			blks[2] = vaesmcq_u8(vaeseq_u8(blks[2], vreinterpretq_u8_m128i(aes_key[0].rd_key[i])));
			blks[1] = vaesmcq_u8(vaeseq_u8(blks[1], vreinterpretq_u8_m128i(aes_key[1].rd_key[i])));
			blks[3] = vaesmcq_u8(vaeseq_u8(blks[3], vreinterpretq_u8_m128i(aes_key[1].rd_key[i])));
		}
	}

	inline void permute_block_2blks(block *_blks) {
		uint8x16_t * blks = (uint8x16_t*)(_blks);
		for (unsigned int i = 0; i < 10; ++i) {
			blks[0] = vaesmcq_u8(vaeseq_u8(blks[2], vreinterpretq_u8_m128i(aes_key[0].rd_key[i])));
			blks[1] = vaesmcq_u8(vaeseq_u8(blks[1], vreinterpretq_u8_m128i(aes_key[1].rd_key[i])));
		}
	}
#endif
};
#endif
