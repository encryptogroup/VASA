#ifndef LIBGARBLE_VAES_H
#define LIBGARBLE_VAES_H

#include <immintrin.h>
#include <cstddef>
#include "block.h"

typedef unsigned char u8;

namespace sci
{
#ifdef __GNUC__
#ifndef __clang__
#pragma GCC push_options
#pragma GCC optimize ("unroll-loops")
#endif
#endif

    // straight ECB using VAES
    template<size_t rounds, size_t width>
    static inline void
        __attribute__((target("avx512f,vaes")))
        VAES_ecb_encrypt_blks(block128* blks, unsigned int nblks, const block128* key)
    {
        static_assert(width % 4 == 0, "width must be a multiple of 4");
        assert(nblks % width == 0);
        constexpr size_t num_registers = width / 4;

        __m512i keys[rounds + 1];
        __m512i data[num_registers];
        // same key for all four lanes
        for (size_t i = 0; i <= rounds; ++i)
            keys[i] = _mm512_broadcast_i32x4(key[i]);

        // progress holds the number of blocks processed
        for (size_t progress = 0; progress < nblks; progress += width)
        {
            // w is always the register index
            for (size_t w = 0; w < num_registers; ++w)
            {
                data[w] = _mm512_loadu_si512(blks + progress + 4 * w);
                data[w] = _mm512_xor_si512(data[w], keys[0]);
            }

            for (size_t r = 1; r < rounds; ++r)
                for (size_t w = 0; w < num_registers; ++w)
                    data[w] = _mm512_aesenc_epi128(data[w], keys[r]);

            for (size_t w = 0; w < num_registers; ++w)
            {
                data[w] = _mm512_aesenclast_epi128(data[w], keys[rounds]);
                _mm512_storeu_si512(blks + progress + 4 * w, data[w]);
            }
        }
    }


    // straight CTR using VAES
    template<size_t rounds, size_t width>
    static inline void
        __attribute__((target("avx512f,vaes")))
        VAES_ctr_encrypt_blks(block128* blks, unsigned int nblks, const block128* key, uint64_t ctr)
    {
        static_assert(width % 4 == 0, "width must be a multiple of 4");
        assert(nblks % width == 0);
        constexpr size_t num_registers = width / 4;

        __m512i keys[rounds + 1];
        __m512i data[num_registers];
        // same key for all four lanes
        for (size_t i = 0; i <= rounds; ++i)
            keys[i] = _mm512_broadcast_i32x4(key[i]);

        __m512i counter = _mm512_set_epi64(0, ctr + 3, 0, ctr + 2, 0, ctr + 1, 0, ctr);
        // we process four blocks per register, so need to increment by four
        __m512i offset = _mm512_set_epi64(0, 4, 0, 4, 0, 4, 0, 4);

        // progress holds the number of blocks processed
        for (size_t progress = 0; progress < nblks; progress += width)
        {
            // w is always the register index
            for (size_t w = 0; w < num_registers; ++w)
            {
                data[w] = counter;
                counter = _mm512_add_epi64(counter, offset);
                data[w] = _mm512_xor_si512(data[w], keys[0]);
            }

            for (size_t r = 1; r < rounds; ++r)
                for (size_t w = 0; w < num_registers; ++w)
                    data[w] = _mm512_aesenc_epi128(data[w], keys[r]);

            for (size_t w = 0; w < num_registers; ++w)
            {
                data[w] = _mm512_aesenclast_epi128(data[w], keys[rounds]);
                _mm512_storeu_si512(blks + progress + 4 * w, data[w]);
            }
        }
    }

    // use the round number as a dynamic parameter
    // statically dispatching to the right implementation
    template<size_t width>
    static inline void
        __attribute__((target("avx512f,vaes")))
        VAES_ecb_encrypt_blks_dyn(block128* blks, unsigned int nblks, const block128* key, const size_t num_rounds)
    {
        switch (num_rounds) {
        case 10:
            VAES_ecb_encrypt_blks<10, width>(blks, nblks, key);
            break;
        case 12:
            VAES_ecb_encrypt_blks<12, width>(blks, nblks, key);
            break;
        case 14:
            VAES_ecb_encrypt_blks<14, width>(blks, nblks, key);
            break;
        default:
            __builtin_unreachable();
            break;
        }
    }

    // use the round number as a dynamic parameter
    // statically dispatching to the right implementation
    template<size_t width>
    static inline void
        __attribute__((target("avx512f,vaes")))
        VAES_ctr_encrypt_blks_dyn(block128* blks, unsigned int nblks, const block128* key, uint64_t ctr, size_t num_rounds)
    {
        switch (num_rounds) {
        case 10:
            VAES_ctr_encrypt_blks<10, width>(blks, nblks, key, ctr);
            break;
        case 12:
            VAES_ctr_encrypt_blks<12, width>(blks, nblks, key, ctr);
            break;
        case 14:
            VAES_ctr_encrypt_blks<14, width>(blks, nblks, key, ctr);
            break;
        default:
            __builtin_unreachable();
            break;
        }
    }

    // this uses an AVX512 / VAES adoption of Gueron's fast key expansion for the fast key expansion
    // https://eprint.iacr.org/2015/751
    template<size_t width>
    static inline __attribute__((target("avx512f,vaes,avx512bw")))    
    void VAES256_KS_ENC_ONE(block128* BLKS, const block128* key) {
        static_assert(width%4==0, "width must be multiple of 4 for VAES");
        constexpr size_t num_registers = width / 4;
        __m512i even_round_key[num_registers], odd_round_key[num_registers], data[num_registers];
        const __m512i first_shuffle_constant = _mm512_set1_epi32(0x0c0f0e0d);
        __m512i rcon = _mm512_set1_epi32(1);
        const __m512i second_shuffle_constant = _mm512_broadcast_i32x4(_mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, 4, 5, 6, 7, 4, 5, 6, 7));
        const __m512i zero_reg = _mm512_setzero_si512();

        const __m512i even_extractor = _mm512_set_epi64(13,12,9,8,5,4,1,0);
        const __m512i odd_extractor = _mm512_set_epi64(15,14,11,10,7,6,3,2);
        const __m512i ONE = _mm512_set_epi64(0, 1, 0, 1, 0, 1, 0, 1);

        for (size_t w = 0; w < num_registers; ++w)
        {
            // this loads the round key for round 0 in lane 0 and for round 1 in lane 1
            // but for both for data word 0
            // therefore we need to use some swap tricks
            const __m512i firstBatch = _mm512_loadu_si512(key + 8*w);
            const __m512i secondBatch = _mm512_loadu_si512(key + 4+8*w);

            even_round_key[w] = _mm512_permutex2var_epi64(firstBatch, even_extractor, secondBatch);
            odd_round_key[w] = _mm512_permutex2var_epi64(firstBatch, odd_extractor, secondBatch);

            data[w] = ONE;
            data[w] = _mm512_xor_si512(data[w], even_round_key[w]);
            data[w] = _mm512_aesenc_epi128(data[w], odd_round_key[w]);
        }
        
        
        for (int r = 0; r < 6; r++)
        {
            for (size_t w = 0; w < num_registers; ++w)
            {
                __m512i tmp2, tmp4;
                tmp2 = _mm512_shuffle_epi8(odd_round_key[w], first_shuffle_constant);
                tmp2 = _mm512_aesenclast_epi128(tmp2, rcon);    
                tmp4 = _mm512_slli_epi64(even_round_key[w], 32);
                even_round_key[w] = _mm512_xor_si512(even_round_key[w], tmp4);
                tmp4 = _mm512_shuffle_epi8(even_round_key[w], second_shuffle_constant);
                even_round_key[w] = _mm512_xor_si512(even_round_key[w], tmp4);
                even_round_key[w] = _mm512_xor_si512(even_round_key[w], tmp2);
                data[w] = _mm512_aesenc_epi128(data[w], even_round_key[w]);
            }
            rcon = _mm512_slli_epi32(rcon, 1);

            for (size_t w = 0; w < num_registers; ++w)
            {
                __m512i tmp2, tmp4;
                tmp2 = _mm512_shuffle_epi32(even_round_key[w], 0xff);
                tmp2 = _mm512_aesenclast_epi128(tmp2, zero_reg);
                tmp4 = _mm512_slli_epi64(odd_round_key[w], 32);
                odd_round_key[w] = _mm512_xor_si512(tmp4, odd_round_key[w]);
                tmp4 = _mm512_shuffle_epi8(odd_round_key[w], second_shuffle_constant);
                odd_round_key[w] = _mm512_xor_si512(tmp4, odd_round_key[w]);
                odd_round_key[w] = _mm512_xor_si512(tmp2, odd_round_key[w]);
                data[w] = _mm512_aesenc_epi128(data[w], odd_round_key[w]);
            }

        }
        for (size_t w = 0; w < num_registers; ++w)
        {
            __m512i tmp2, tmp4;
            tmp2 = _mm512_shuffle_epi8(odd_round_key[w], first_shuffle_constant);
            tmp2 = _mm512_aesenclast_epi128(tmp2, rcon);
            tmp4 = _mm512_slli_epi64(even_round_key[w], 32);
            even_round_key[w] = _mm512_xor_si512(even_round_key[w], tmp4);
            tmp4 = _mm512_shuffle_epi8(even_round_key[w], second_shuffle_constant);
            even_round_key[w] = _mm512_xor_si512(even_round_key[w], tmp4);
            even_round_key[w] = _mm512_xor_si512(even_round_key[w], tmp2);
            data[w] = _mm512_aesenclast_epi128(data[w], even_round_key[w]);
            _mm512_storeu_si512(BLKS+4*w, data[w]);
        }
    }

#ifdef __GNUC__
#ifndef __clang__
#pragma GCC pop_options
#endif
#endif
}

#endif