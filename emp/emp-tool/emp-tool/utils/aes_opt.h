#ifndef EMP_AES_OPT_KS_H__
#define EMP_AES_OPT_KS_H__

#include "emp-tool/utils/aes.h"

namespace emp {
template<int NumKeys>
static inline void ks_rounds(block* keys, block con, block con3, block mask, int r) {
	for (int i = 0; i < NumKeys; ++i) {
		block key = keys[(r-1)*NumKeys+i];
		block x2 =_mm_shuffle_epi8(key, mask);
		block aux = _mm_aesenclast_si128 (x2, con);

		block globAux=_mm_slli_epi64(key, 32);
		key=_mm_xor_si128(globAux, key);
		globAux=_mm_shuffle_epi8(key, con3);
		key=_mm_xor_si128(globAux, key);
		keys[r * NumKeys + i] = _mm_xor_si128(aux, key);
	}
}

template<int NumRegisters> __attribute__((target("avx512f,vaes,avx512bw")))
static inline void ks_rounds_vaes(__m512i keys[NumRegisters], __m512i rcon, __m512i con3, __m512i mask, int r) {
	for (int i = 0; i < NumRegisters; ++i) {
		__m512i x2 = _mm512_shuffle_epi8(keys[i], mask);
		__m512i aux = _mm512_aesenclast_epi128(x2, rcon);

		__m512i globAux = _mm512_slli_epi64(keys[i], 32);
		keys[i] = _mm512_xor_si512(globAux, keys[i]);
		globAux = _mm512_shuffle_epi8(keys[i], con3);
		keys[i] = _mm512_xor_si512(globAux, keys[i]);
	}
}

/*
 * AES key scheduling for 8 keys
 * [REF] Implementation of "Fast Garbling of Circuits Under Standard Assumptions"
 * https://eprint.iacr.org/2015/751.pdf
 */
template<int NumKeys>
static inline void AES_opt_key_schedule_aesni(block* user_key, block*keys) {
	block con = _mm_set_epi32(1,1,1,1);
	block con2 = _mm_set_epi32(0x1b,0x1b,0x1b,0x1b);
	block con3 = _mm_set_epi32(0x07060504,0x07060504,0x0ffffffff,0x0ffffffff);
	block mask = _mm_set_epi32(0x0c0f0e0d,0x0c0f0e0d,0x0c0f0e0d,0x0c0f0e0d);

	for(int i = 0; i < NumKeys; ++i) {
		keys[i] = user_key[i];
	}

	ks_rounds<NumKeys>(keys, con, con3, mask, 1);
	con=_mm_slli_epi32(con, 1);
	ks_rounds<NumKeys>(keys, con, con3, mask, 2);
	con=_mm_slli_epi32(con, 1);
	ks_rounds<NumKeys>(keys, con, con3, mask, 3);
	con=_mm_slli_epi32(con, 1);
	ks_rounds<NumKeys>(keys, con, con3, mask, 4);
	con=_mm_slli_epi32(con, 1);
	ks_rounds<NumKeys>(keys, con, con3, mask, 5);
	con=_mm_slli_epi32(con, 1);
	ks_rounds<NumKeys>(keys, con, con3, mask, 6);
	con=_mm_slli_epi32(con, 1);
	ks_rounds<NumKeys>(keys, con, con3, mask, 7);
	con=_mm_slli_epi32(con, 1);
	ks_rounds<NumKeys>(keys, con, con3, mask, 8);
	con=_mm_slli_epi32(con, 1);
	ks_rounds<NumKeys>(keys, con2, con3, mask, 9);
	con2=_mm_slli_epi32(con2, 1);
	ks_rounds<NumKeys>(keys, con2, con3, mask, 10);
}

// This uses GFNI for the RCON multiplications
// instead of shifts + sets as usual
// otherwise it's very similar to the above AES-NI
// translated to a width of 4
template<int NumKeys> __attribute__((target("avx512f,gfni")))
static inline void AES_opt_key_schedule_vaes(block* user_key, block* keys) {
	static_assert(NumKeys%4==0, "VAES only supports multiples of 4");
	block con3 = _mm_set_epi32(0x07060504, 0x07060504, 0x0ffffffff, 0x0ffffffff);
	__m512i wideShuffle = _mm512_broadcast_i32x4(con3);
	block mask = _mm_set_epi32(0x0c0f0e0d, 0x0c0f0e0d, 0x0c0f0e0d, 0x0c0f0e0d);
	__m512i wideMask = _mm512_broadcast_i32x4(mask);

	constexpr int numRegisters = NumKeys / 4;

	__m512i current_keys[numRegisters];

	for (int i = 0; i < numRegisters; ++i) {
		current_keys[i] = _mm512_loadu_si512(reinterpret_cast<__m512i*>(&user_key[4 * i]));
		_mm512_storeu_si512(reinterpret_cast<__m512i*>(&keys[4 * i]), current_keys[i]);
	}

	__m512i rcon = _mm512_set1_epi8(1);
	__m512i rcon_multiplier = _mm512_set1_epi8(2);

	for (int r = 1; r <= 10; ++r) {
		ks_rounds_vaes<numRegisters>(current_keys, rcon, wideShuffle, wideMask, r);
		rcon = _mm512_gf2p8mul_epi8(rcon, rcon_multiplier);
		for(int i=0;i<numRegisters;++i)
			_mm512_storeu_si512(reinterpret_cast<__m512i*>(&keys[4 * i+r*NumKeys]), current_keys[i]);
	}
}

// only use VAES if we have enough key schedules
// and don't get leftovers
template<int NumKeys>
static inline void AES_opt_key_schedule(block* user_key, block* keys) {
	if constexpr (NumKeys >= 8 && NumKeys % 4 == 0)
		AES_opt_key_schedule_vaes<NumKeys>(user_key, keys);
	else
		AES_opt_key_schedule_aesni<NumKeys>(user_key, keys);
}


/*
 * With numKeys keys, use each key to encrypt numEncs blocks.
 */
#ifdef __x86_64__
template<int numKeysPossible, int numKeys, int numEncs>
static inline void ParaEncAESNI(block *blks, __m128i*keys, int key_offset) {
	block * first = blks;
	for(size_t i = 0; i < numKeys; ++i) {
		block K = keys[key_offset + i];
		for(size_t j = 0; j < numEncs; ++j) {
			*blks = *blks ^ K;
			++blks;
		}
	}

	for (unsigned int r = 1; r < 10; ++r) { 
		blks = first;
		for(size_t i = 0; i < numKeys; ++i) {
			block K = keys[key_offset + i+r* numKeysPossible];
			for(size_t j = 0; j < numEncs; ++j) {
				*blks = _mm_aesenc_si128(*blks, K);
				++blks;
			}
		}
	}

	blks = first;
	for(size_t i = 0; i < numKeys; ++i) {
		block K = keys[key_offset + i+10 * numKeysPossible];
		for(size_t j = 0; j < numEncs; ++j) {
			*blks = _mm_aesenclast_si128(*blks, K);
			++blks;
		}
	}
}

// this is a straight translation of the AES-NI version
// to 4-width VAES while more explicitly using intrinsics
// and not relying on the extension that adds operators to __mXXXi
// also uses gather and scatter instructions to interact with the memory
// instead of loading 128-bit blocks and combining them
template<int numKeysPossible, int numKeys, int numEncs>
static inline void ParaEncVAES(block* blks, __m128i* keys, int key_offset) {
	static_assert(numKeys % 4 == 0, "only multiples of 4 are supported for VAES");
	static_assert(numEncs <= 2, "Only a multiplier of 1 and 2 is currently supported");

	constexpr int numKeyRegisters = numKeys / 4;

	__m512i data[numKeyRegisters * numEncs];
	const __m512i base_indices = _mm512_set_epi64(numEncs * 6 + 1, numEncs * 6, numEncs * 4 + 1, numEncs * 4, numEncs * 2 + 1, numEncs * 2, 1, 0);
	const __m512i base_offset = _mm512_set1_epi64(2);


	if constexpr (numEncs == 1) {
		for (size_t i = 0; i < numKeyRegisters; ++i) {
			data[i] = _mm512_loadu_si512(reinterpret_cast<__m512i*>(&blks[4 * i]));
		}
	}
	else {
		for (size_t i = 0; i < numKeyRegisters; ++i) {
			__m512i indices = base_indices;
			for (size_t j = 0; j < numEncs; ++j) {
				data[i * numEncs + j] = _mm512_i64gather_epi64(indices, reinterpret_cast<__m512i*>(&blks[4 *numEncs* i]),8);
				indices = _mm512_add_epi64(indices, base_offset);
			}
		}
	}

	for (size_t i = 0; i < numKeyRegisters; ++i) {
		const __m512i currentKeys = _mm512_loadu_si512(reinterpret_cast<__m512i*>(&keys[key_offset + 4 * i + 0 * numKeysPossible]));
		for (size_t j = 0; j < numEncs; ++j) {
			data[i * numEncs + j] = _mm512_xor_si512(data[i * numEncs + j], currentKeys);
		}
	}

	for (unsigned int r = 1; r < 10; ++r) {
		for (size_t i = 0; i < numKeyRegisters; ++i) {
			const __m512i currentKeys = _mm512_loadu_si512(reinterpret_cast<__m512i*>(&keys[key_offset + 4 * i + r * numKeysPossible]));
			for (size_t j = 0; j < numEncs; ++j) {
				data[i * numEncs + j] = _mm512_aesenc_epi128(data[i * numEncs + j], currentKeys);
			}
		}
	}

	for (size_t i = 0; i < numKeyRegisters; ++i) {
		const __m512i currentKeys = _mm512_loadu_si512(reinterpret_cast<__m512i*>(&keys[key_offset + 4 * i + 10 * numKeysPossible]));
		for (size_t j = 0; j < numEncs; ++j) {
			data[i * numEncs + j] = _mm512_aesenclast_epi128(data[i * numEncs + j], currentKeys);
		}
	}

	if constexpr (numEncs == 1) {
		for (size_t i = 0; i < numKeyRegisters; ++i) {
			_mm512_storeu_si512(reinterpret_cast<__m512i*>(&blks[4 * i]), data[i]);
		}
	}
	else {
		for (size_t i = 0; i < numKeyRegisters; ++i) {
			__m512i indices = base_indices;
			for (size_t j = 0; j < numEncs; ++j) {
				_mm512_i64scatter_epi64(reinterpret_cast<__m512i*>(&blks[4*numEncs * i]), indices, data[i * numEncs + j],8);
				indices = _mm512_add_epi64(indices, base_offset);
			}
		}
	}
}

// similar distribution reasoning as for the key schedule
template<int numKeysPossible, int numKeys, int numEncs>
static inline void ParaEnc(block* blks, __m128i* keys, int key_offset) {
	if constexpr (numKeys % 4 == 0 && numKeys >= 8)
		ParaEncVAES<numKeysPossible,numKeys, numEncs>(blks, keys, key_offset);
	else
		ParaEncAESNI<numKeysPossible,numKeys, numEncs>(blks, keys, key_offset);
}

#elif __aarch64__
template<int numKeys, int numEncs>
static inline void ParaEnc(block *_blks, AES_KEY *keys) {
	uint8x16_t * first = (uint8x16_t*)(_blks);

	for (unsigned int r = 0; r < 9; ++r) { 
		auto blks = first;
		for(size_t i = 0; i < numKeys; ++i) {
			uint8x16_t K = vreinterpretq_u8_m128i(keys[i].rd_key[r]);
			for(size_t j = 0; j < numEncs; ++j, ++blks)
			   *blks = vaeseq_u8(*blks, K);
		}
		blks = first;
		for(size_t i = 0; i < numKeys; ++i) {
			for(size_t j = 0; j < numEncs; ++j, ++blks)
			   *blks = vaesmcq_u8(*blks);
		}
	}
	
	auto blks = first;
	for(size_t i = 0; i < numKeys; ++i) {
		uint8x16_t K = vreinterpretq_u8_m128i(keys[i].rd_key[9]);
		for(size_t j = 0; j < numEncs; ++j, ++blks)
			*blks = vaeseq_u8(*blks, K) ^ K;
	}

}
#endif

}
#endif
