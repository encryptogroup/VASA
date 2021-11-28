/* crypto/aes/aes.h -*- mode:C; c-file-style: "eay" -*- */
/* ====================================================================
 * Copyright (c) 1998-2002 The OpenSSL Project.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in
 *    the documentation and/or other materials provided with the
 *    distribution.
 *
 * 3. All advertising materials mentioning features or use of this
 *    software must display the following acknowledgment:
 *    "This product includes software developed by the OpenSSL Project
 *    for use in the OpenSSL Toolkit. (http://www.openssl.org/)"
 *
 * 4. The names "OpenSSL Toolkit" and "OpenSSL Project" must not be used to
 *    endorse or promote products derived from this software without
 *    prior written permission. For written permission, please contact
 *    openssl-core@openssl.org.
 *
 * 5. Products derived from this software may not be called "OpenSSL"
 *    nor may "OpenSSL" appear in their names without prior written
 *    permission of the OpenSSL Project.
 *
 * 6. Redistributions of any form whatsoever must retain the following
 *    acknowledgment:
 *    "This product includes software developed by the OpenSSL Project
 *    for use in the OpenSSL Toolkit (http://www.openssl.org/)"
 *
 * THIS SOFTWARE IS PROVIDED BY THE OpenSSL PROJECT ``AS IS'' AND ANY
 * EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE OpenSSL PROJECT OR
 * ITS CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
 * OF THE POSSIBILITY OF SUCH DAMAGE.
 * ====================================================================
 *
 */

#ifndef EMP_AES_H
#define EMP_AES_H

#include "emp-tool/utils/block.h"
#include <immintrin.h>

namespace emp {

	typedef struct { block rd_key[11]; unsigned int rounds; } AES_KEY;

#define EXPAND_ASSIST(v1,v2,v3,v4,shuff_const,aes_const)                    \
    v2 = _mm_aeskeygenassist_si128(v4,aes_const);                           \
    v3 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(v3),              \
                                         _mm_castsi128_ps(v1), 16));        \
    v1 = _mm_xor_si128(v1,v3);                                              \
    v3 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(v3),              \
                                         _mm_castsi128_ps(v1), 140));       \
    v1 = _mm_xor_si128(v1,v3);                                              \
    v2 = _mm_shuffle_epi32(v2,shuff_const);                                 \
    v1 = _mm_xor_si128(v1,v2)

	inline void
#ifdef __x86_64__
		__attribute__((target("aes,sse2")))
#endif
		AES_set_encrypt_key(const block userkey, AES_KEY* key) {
		block x0, x1, x2;
		block* kp = key->rd_key;
		kp[0] = x0 = userkey;
		x2 = _mm_setzero_si128();
		EXPAND_ASSIST(x0, x1, x2, x0, 255, 1);
		kp[1] = x0;
		EXPAND_ASSIST(x0, x1, x2, x0, 255, 2);
		kp[2] = x0;
		EXPAND_ASSIST(x0, x1, x2, x0, 255, 4);
		kp[3] = x0;
		EXPAND_ASSIST(x0, x1, x2, x0, 255, 8);
		kp[4] = x0;
		EXPAND_ASSIST(x0, x1, x2, x0, 255, 16);
		kp[5] = x0;
		EXPAND_ASSIST(x0, x1, x2, x0, 255, 32);
		kp[6] = x0;
		EXPAND_ASSIST(x0, x1, x2, x0, 255, 64);
		kp[7] = x0;
		EXPAND_ASSIST(x0, x1, x2, x0, 255, 128);
		kp[8] = x0;
		EXPAND_ASSIST(x0, x1, x2, x0, 255, 27);
		kp[9] = x0;
		EXPAND_ASSIST(x0, x1, x2, x0, 255, 54);
		kp[10] = x0;
		key->rounds = 10;
	}

#ifdef __x86_64__
#ifdef __GNUC__
#ifndef __clang__
#pragma GCC push_options
#pragma GCC optimize ("unroll-loops")
#endif
#endif
	template<int width> __attribute__((target("vaes,avx512f")))
	inline void AES_ecb_encrypt_blks_vaes(block* blks, unsigned int nblks, const __m512i key[11]) {
		constexpr int data_width = 4 * width;
		assert(nblks % data_width == 0);
		
		for (size_t i = 0; i < nblks; i += data_width) {
			__m512i batch[width];
			for (size_t w = 0; w < width; ++w)
			{
				batch[w] = _mm512_loadu_si512(reinterpret_cast<__m512i*>(&blks[i + 4 * w]));
				batch[w] = _mm512_xor_si512(batch[w], key[0]);
			}

			for (size_t r = 1; r < 10; ++r)
				for (size_t w = 0; w < width; ++w)
					batch[w] = _mm512_aesenc_epi128(batch[w], key[r]);

			for (size_t w = 0; w < width; ++w) {
				batch[w] = _mm512_aesenclast_epi128(batch[w], key[10]);
				_mm512_storeu_si512(reinterpret_cast<__m512i*>(&blks[i + 4 * w]), batch[w]);
			}
				
		}
	}

	
		template<int width> __attribute__((target("aes,sse2")))
	inline void AES_ecb_encrypt_blks_aesni(block* blks, unsigned int nblks, const AES_KEY* key) {
		assert(nblks % width == 0);

		for (size_t i = 0; i < nblks; i += width) {
			block batch[width];
			for (size_t w = 0; w < width; ++w)
				batch[w] = _mm_xor_si128(blks[i + w], key->rd_key[0]);
			for (size_t r = 1; r < 10; ++r)
				for (size_t w = 0; w < width; ++w)
					batch[w] = _mm_aesenc_si128(batch[w], key->rd_key[r]);
			for (size_t w = 0; w < width; ++w)
				blks[i + w] = _mm_aesenclast_si128(batch[w], key->rd_key[10]);
		}
	}

		// this first computes the batches
		// and then uses 16-block batches with VAES
		// then 4 block batches (this is less instructions than the AES-NI version)
		// then the clean up with AES-NI
		__attribute__((target("avx512f")))
			inline void AES_ecb_encrypt_blks(block* blks, unsigned int nblks, const AES_KEY* key) {
			unsigned int leftovers_for_vaes = nblks % 16;
			nblks -= leftovers_for_vaes;
			unsigned int leftovers_for_aesni = leftovers_for_vaes % 4;
			leftovers_for_vaes -= leftovers_for_aesni;

			__m512i vaes_keys[11];
			for (size_t i = 0; i < 11; ++i)
				vaes_keys[i] = _mm512_broadcast_i32x4(key->rd_key[i]);

			AES_ecb_encrypt_blks_vaes<4>(blks, nblks, vaes_keys);
			AES_ecb_encrypt_blks_vaes<1>(blks + nblks, leftovers_for_vaes, vaes_keys);
			AES_ecb_encrypt_blks_aesni<1>(blks + nblks + leftovers_for_vaes, leftovers_for_aesni, key);
}
#ifdef __GNUC_
#ifndef __clang___
#pragma GCC pop_options
#endif
#endif
#elif __aarch64__
	inline void AES_ecb_encrypt_blks(block* _blks, unsigned int nblks, const AES_KEY* key) {
		uint8x16_t* blks = (uint8x16_t*)(_blks);
		uint8x16_t* keys = (uint8x16_t*)(key->rd_key);
		auto* first = blks;
		for (unsigned int j = 0; j < key->rounds - 1; ++j) {
			uint8x16_t key_j = (uint8x16_t)keys[j];
			blks = first;
			for (unsigned int i = 0; i < nblks; ++i, ++blks)
				*blks = vaeseq_u8(*blks, key_j);
			blks = first;
			for (unsigned int i = 0; i < nblks; ++i, ++blks)
				*blks = vaesmcq_u8(*blks);
		}
		uint8x16_t last_key = (uint8x16_t)keys[key->rounds - 1];
		for (unsigned int i = 0; i < nblks; ++i, ++first)
			*first = vaeseq_u8(*first, last_key) ^ last_key;
	}
#endif

#ifdef __GNUC__
#ifndef __clang__
#pragma GCC push_options
#pragma GCC optimize ("unroll-loops")
#endif
#endif
	template<int N>
	inline void AES_ecb_encrypt_blks(block* blks, const AES_KEY* key) {
		AES_ecb_encrypt_blks(blks, N, key);
	}
#ifdef __GNUC_
#ifndef __clang___
#pragma GCC pop_options
#endif
#endif

	inline void
#ifdef __x86_64__
		__attribute__((target("aes,sse2")))
#endif
		AES_set_decrypt_key_fast(AES_KEY* dkey, const AES_KEY* ekey) {
		int j = 0;
		int i = ekey->rounds;
#if (OCB_KEY_LEN == 0)
		dkey->rounds = i;
#endif
		dkey->rd_key[i--] = ekey->rd_key[j++];
		while (i)
			dkey->rd_key[i--] = _mm_aesimc_si128(ekey->rd_key[j++]);
		dkey->rd_key[i] = ekey->rd_key[j];
	}

	inline void
#ifdef __x86_64__
		__attribute__((target("aes,sse2")))
#endif
		AES_set_decrypt_key(block userkey, AES_KEY* key) {
		AES_KEY temp_key;
		AES_set_encrypt_key(userkey, &temp_key);
		AES_set_decrypt_key_fast(key, &temp_key);
	}

	inline void
#ifdef __x86_64__
		__attribute__((target("aes,sse2")))
#endif
		AES_ecb_decrypt_blks(block* blks, unsigned nblks, const AES_KEY* key) {
		unsigned i, j, rnds = key->rounds;
		for (i = 0; i < nblks; ++i)
			blks[i] = _mm_xor_si128(blks[i], key->rd_key[0]);
		for (j = 1; j < rnds; ++j)
			for (i = 0; i < nblks; ++i)
				blks[i] = _mm_aesdec_si128(blks[i], key->rd_key[j]);
		for (i = 0; i < nblks; ++i)
			blks[i] = _mm_aesdeclast_si128(blks[i], key->rd_key[j]);
	}
}
#endif
