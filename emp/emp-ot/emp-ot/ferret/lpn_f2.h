#ifndef EMP_LPN_F2K_H__
#define EMP_LPN_F2K_H__

#include "emp-tool/emp-tool.h"
using namespace emp;

static inline void PrintKey(__m128i data) {
	uint8_t key[16];
	_mm_storeu_si128((__m128i*)key, data);

	for (uint32_t i = 0; i < 16; i++) {
		std::cout << std::setw(2) << std::setfill('0') << (std::hex) << (uint32_t)key[i];
	}
	std::cout << (std::dec);
}

//Implementation of local linear code on F_2^k
//Performance highly dependent on the CPU cache size
template<int d = 10>
class LpnF2 { public:
	int party;
	int k, n;
	ThreadPool * pool;
	NetIO *io;
	int threads;
	block seed;
	int mask;
	LpnF2 (int party, int n, int k, ThreadPool * pool, NetIO *io, int threads) {
		this->party = party;
		this->k = k;
		this->n = n;
		this->pool = pool;
		this->io = io;
		this->threads = threads;
		mask = 1;
		while(mask < k) {
			mask <<=1;
			mask = mask | 0x1;
		}
	}

	__attribute__((target("avx512f,avx512vl")))
	void __compute4(block * nn, const block * kk, int i, PRP * prp) {
		// block size increased from 1 to 4 to allow 4-wide VAES operations
		constexpr size_t num_blocks = d * 16 / 4;
		block tmp[num_blocks];
		for(int m = 0; m < num_blocks; ++m)
			tmp[m] = makeBlock(i, m);
		AES_ecb_encrypt_blks(tmp, num_blocks, &prp->aes);
		uint32_t* r = (uint32_t*)(tmp);

		// the below was an attempt to leverage
		// AVX512 instead of the general purpose registers
		// for the processing of the below fetch and accumulate loop
		// benchmarks showed reduced performance, so it remained unused
		/*
		__m512i vmask = _mm512_set1_epi32(mask);
		__m512i vk = _mm512_set1_epi32(k);
		__m512i vd = _mm512_set1_epi32(d);
		__m512i base_fetch_offset = _mm512_set_epi32(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0);
		__m512i base_index = _mm512_mullo_epi32(vd, base_fetch_offset);
		__m512i fetcher_offset = _mm512_set_epi32(3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0);
		__m512i increment_offset = _mm512_set1_epi32(1);
		__m512i permutation_indices = _mm512_set_epi32(3,3,3,3,2,2,2,2,1,1,1,1,0,0,0,0);

		__m512i data[4];
		for(size_t m=0;m<4;++m)
			data[m] = _mm512_loadu_si512((__m512i*)(&nn[i + 4 * m]));

		__m512i r_indices = base_index;


		for (int j = 0; j < d; ++j) {
			__m512i gathered_rs = _mm512_i32gather_epi32(r_indices,tmp, 4);
			__m512i masked_rs = _mm512_and_epi32(gathered_rs, vmask);
			__mmask16 cmp_results = _mm512_cmp_epi32_mask(masked_rs, vk, _MM_CMPINT_NLT);

			__m512i reduced_rs = _mm512_mask_sub_epi32(masked_rs, cmp_results, masked_rs, vk);

			__m512i scaled_kk_indices = _mm512_slli_epi32(reduced_rs, 2);

			__m128i kk_indices_small[4];
			kk_indices_small[0] = _mm512_extracti32x4_epi32(scaled_kk_indices, 0);
			kk_indices_small[1] = _mm512_extracti32x4_epi32(scaled_kk_indices, 1);
			kk_indices_small[2] = _mm512_extracti32x4_epi32(scaled_kk_indices, 2);
			kk_indices_small[3] = _mm512_extracti32x4_epi32(scaled_kk_indices, 3);

			for (size_t m = 0; m < 4; ++m) {
				__m512i kk_indices = _mm512_broadcast_i32x4(kk_indices_small[m]);
				kk_indices = _mm512_permutexvar_epi32(permutation_indices, kk_indices);
				kk_indices = _mm512_add_epi32(kk_indices, fetcher_offset);
				__m512i mask = _mm512_i32gather_epi32(kk_indices, kk, 4);
				data[m] = _mm512_xor_epi32(data[m], mask);
			}

			r_indices = _mm512_add_epi32(r_indices, increment_offset);
		}

		for (size_t m = 0; m < 4; ++m)
			_mm512_storeu_si512((__m512i*)(&nn[i + 4 * m]), data[m]);*/

		
	
		for (int m = 0; m < 16; ++m) {
			for (int j = 0; j < d; ++j) {
				int index = (*r) & mask;
				++r;
				index = index >= k ? index - k : index;
				nn[i + m] = nn[i + m] ^ kk[index];
			}
		}
	}

	void __compute1(block * nn, const block * kk, int i, PRP*prp) {
		block tmp[3];
		for(int m = 0; m < 3; ++m)
			tmp[m] = makeBlock(i, m);
		prp->permute_block(tmp, 3);
		uint32_t* r = (uint32_t*)(tmp);
		for (int j = 0; j < d; ++j)
			nn[i] = nn[i] ^ kk[r[j]%k];
	}

	void task(block * nn, const block * kk, int start, int end) {
		PRP prp(seed);
		int j = start;
		for(; j < end-16; j+=16)
			__compute4(nn, kk, j, &prp);
		for(; j < end; ++j)
			__compute1(nn, kk, j, &prp);
	}

	void compute(block * nn, const block * kk) {
		vector<std::future<void>> fut;
		int width = n/threads;
		seed = seed_gen();
		for(int i = 0; i < threads - 1; ++i) {
			int start = i * width;
			int end = min((i+1)* width, n);
			fut.push_back(pool->enqueue([this, nn, kk, start, end]() {
				task(nn, kk, start, end);
			}));
		}
		int start = (threads - 1) * width;
		int end = min(threads * width, n);
		task(nn, kk, start, end);

		for (auto &f: fut) f.get();
	}

	block seed_gen() {
		block seed;
		if(party == ALICE) {
			PRG prg;
			prg.random_block(&seed, 1);
			io->send_data(&seed, sizeof(block));
		} else {
			io->recv_data(&seed, sizeof(block));
		}io->flush();
		return seed;
	}
	void bench(block * nn, const block * kk) {
		vector<std::future<void>> fut;
		int width = n/threads;
		for(int i = 0; i < threads - 1; ++i) {
			int start = i * width;
			int end = min((i+1)* width, n);
			fut.push_back(pool->enqueue([this, nn, kk, start, end]() {
				task(nn, kk, start, end);
			}));
		}
		int start = (threads - 1) * width;
		int end = min(threads * width, n);
		task(nn, kk, start, end);

		for (auto &f: fut) f.get();
	}

};
#endif
