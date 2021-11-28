//TODO: check MACs
#ifndef FPRE_MP_H__
#define FPRE_MP_H__
#include <emp-tool/emp-tool.h>
#include <thread>
#include "abitmp.h"
#include "netmp.h"
#include "cmpc_config.h"

constexpr int HNID_BATCH_SIZE = 8;
constexpr int RECV_BATCH_SIZE = 16;
constexpr int EVAL_BATCH_SIZE = 16;
constexpr int GARB_BATCH_SIZE = 4;

using namespace emp;
template<int nP>
class FpreMP { public:
	
	ThreadPool *pool;
	int party;
	NetIOMP<nP> * io;
	ABitMP<nP>* abit;
	block Delta;
	CRH * prps;
	CRH * prps2;
	PRG * prgs;
	PRG prg;
	int ssp;
	FpreMP(NetIOMP<nP> * io[2], ThreadPool * pool, int party, int ssp = 40) {
		this->party = party;
		this->pool = pool;
		this->io = io[0];
		this ->ssp = ssp;
		abit = new ABitMP<nP>(io[1], pool, party);
		Delta = abit->Delta;
		prps = new CRH[nP+1];
		prps2 = new CRH[nP+1];
		prgs = new PRG[nP+1];
	}
	~FpreMP(){
		delete[] prps;
		delete[] prps2;
		delete[] prgs;
		delete abit;
	}
	int get_bucket_size(int size) {
		size = max(size, 320);
		int batch_size = ((size+2-1)/2)*2;
		if(batch_size >= 280*1000)
			return 3;
		else if(batch_size >= 3100)
			return 4;
		else return 5;
	}
	void compute(block * MAC[nP+1], block * KEY[nP+1], bool * r, int length) {
		int bucket_size = get_bucket_size(length);
		block * tMAC[nP+1];
		block * tKEY[nP+1];
		block * tKEYphi[nP+1];
		block * tMACphi[nP+1];
		block * phi;
		block *X [nP+1];
		bool *tr = new bool[length*bucket_size*3+3*ssp];
		phi = new block[length*bucket_size];
		bool *s[nP+1], *e = new bool[length*bucket_size];
		for(int i = 1; i <= nP; ++i) {
			tMAC[i] = new block[length*bucket_size*3+3*ssp];
			tKEY[i] = new block[length*bucket_size*3+3*ssp];
			tKEYphi[i] = new block[length*bucket_size+3*ssp];
			tMACphi[i] = new block[length*bucket_size+3*ssp];
			X[i] = new block[ssp];
		}
		for(int i = 0; i <= nP; ++i) {
			s[i] = new bool[length*bucket_size];
			memset(s[i], 0, length*bucket_size);
		}
		prg.random_bool(tr, length*bucket_size*3+3*ssp);
		// memset(tr, false, length*bucket_size*3+3*ssp);
		abit->compute(tMAC, tKEY, tr, length*bucket_size*3 + 3*ssp);
		vector<future<void>>	 res;

		for(int i = 1; i <= nP; ++i) for(int j = 1; j <= nP; ++j) if (i < j ) {
			if(i == party) {
				res.push_back(pool->enqueue([this, tKEY, tr, s, length, bucket_size, j]() {
					prgs[j].random_bool(s[j], length*bucket_size);
					uint8_t data[GARB_BATCH_SIZE];
					for(int k = 0; k < length*bucket_size;) {
						const int current_batch = std::min(GARB_BATCH_SIZE, length * bucket_size - k);
						// In this loop and the one a dozen lines below
						// we changed the garble / evaluate operations
						// to take multiple tasks at once
						// with the above code to ensure that we hit correct sizes
						// and evaluate early
						garble(tKEY[j], tr, s[j], k, j, current_batch, data);
						io->send_data(j, &data, current_batch);
						for(int kk=0;kk<current_batch;++kk)
							s[j][(k + kk)] = (s[j][(k + kk)] != (tr[3* (k + kk)] and tr[3* (k + kk) +1]));
						k += current_batch;
					}
					io->flush(j);
				}));
			} else if (j == party) {
				res.push_back(pool->enqueue([this, tMAC, tr, s, length, bucket_size, i]() {
					for(int k = 0; k < length*bucket_size;) {
						const int current_batch = std::min(EVAL_BATCH_SIZE, length * bucket_size - k);
						uint8_t data[EVAL_BATCH_SIZE] = {0};
						bool outtmp[EVAL_BATCH_SIZE];
						io->recv_data(i, data, current_batch);
						// see above for explanation
						evaluate(data, tMAC[i], tr, k, i, current_batch, outtmp);
						for(int kk=0;kk<current_batch;++kk)
							s[i][k+kk] = (outtmp[kk] != (tr[3*(k + kk)] and tr[3* (k + kk) +1]));
						k += current_batch;
					}
				}));
			}
		}
		joinNclean(res);
		for(int k = 0; k < length*bucket_size; ++k) {
			s[0][k] = (tr[3*k] and tr[3*k+1]);
			for(int i = 1; i <= nP; ++i) 
				if (i != party) {
					s[0][k] = (s[0][k] != s[i][k]);
				}
			e[k] = (s[0][k] != tr[3*k+2]);
			tr[3*k+2] = s[0][k];
		}

#ifdef __debug
		check_correctness(io, tr, length*bucket_size, party);
#endif
		for(int i = 1; i <= nP; ++i) for(int j = 1; j<= nP; ++j) if( (i < j) and (i == party or j == party) ) {
			int party2 = i + j - party;
			res.push_back(pool->enqueue([this, e, length, bucket_size, party2]() {
				io->send_data(party2, e, length*bucket_size);
				io->flush(party2);
			}));
			res.push_back(pool->enqueue([this, tKEY, length, bucket_size, party2]() {
				bool * tmp = new bool[length*bucket_size];
				io->recv_data(party2, tmp, length*bucket_size);
				for(int k = 0; k < length*bucket_size; ++k) {
					if(tmp[k])
						tKEY[party2][3*k+2] = tKEY[party2][3*k+2] ^ Delta;
				}
				delete[] tmp;
			}));
		}
		joinNclean(res);
#ifdef __debug
		check_MAC(io, tMAC, tKEY, tr, Delta, length*bucket_size*3, party);
#endif		
		auto ret = abit->check(tMAC, tKEY, tr, length*bucket_size*3 + 3*ssp);
		ret.get();
		//check compute phi
		for(int k = 0; k < length*bucket_size; ++k) {
			phi[k] = zero_block;
			for(int i = 1; i <= nP; ++i) if (i != party) {
				phi[k] = phi[k] ^ tKEY[i][3*k+1];
				phi[k] = phi[k] ^ tMAC[i][3*k+1];
			}
			if(tr[3*k+1])phi[k] = phi[k] ^ Delta;
		}

		for(int i = 1; i <= nP; ++i) for(int j = 1; j<= nP; ++j) if( (i < j) and (i == party or j == party) ) {
			int party2 = i + j - party;
			res.push_back(pool->enqueue([this, tKEY, tKEYphi, phi, length, bucket_size, party2]() {
				// TODO use larger batches here
				for(int k = 0; k < length*bucket_size;) {
					HnIDProcess(tKEY, tKEYphi, phi, length * bucket_size, party2, k);
				}
				io->flush(party2);
			}));
			res.push_back(pool->enqueue([this, tMAC, tMACphi, tr, length, bucket_size, party2]() {
				// TODO use larger batches here (with H template variant)
				for(int k = 0; k < length*bucket_size;) {
					RecvProcess(tMAC, tMACphi, tr, length* bucket_size, party2,k);
				}
			}));
		}
		joinNclean(res);

		bool * xs = new bool[length*bucket_size];
		for(int i = 0; i < length*bucket_size; ++i) xs[i] = tr[3*i];

#ifdef __debug
		check_MAC_phi(tMACphi, tKEYphi, phi, xs, length*bucket_size);
#endif
		//tKEYphti use as H
		for(int k = 0; k < length*bucket_size; ++k) {
			tKEYphi[party][k] = zero_block;
			for(int i = 1; i <= nP; ++i) if (i != party) {
				tKEYphi[party][k] = tKEYphi[party][k] ^ tKEYphi[i][k];
				tKEYphi[party][k] = tKEYphi[party][k] ^ tMACphi[i][k];
				tKEYphi[party][k] = tKEYphi[party][k] ^ tKEY[i][3*k+2];
				tKEYphi[party][k] = tKEYphi[party][k] ^ tMAC[i][3*k+2];
			}
			if(tr[3*k])	 tKEYphi[party][k] = tKEYphi[party][k] ^ phi[k];
			if(tr[3*k+2])tKEYphi[party][k] = tKEYphi[party][k] ^ Delta;
		}

#ifdef __debug
		check_zero(tKEYphi[party], length*bucket_size);
#endif

		PRG prgf(fix_key);
		char (*dgst)[Hash::DIGEST_SIZE] = new char[nP+1][Hash::DIGEST_SIZE];
		bool * tmp = new bool[length*bucket_size];
		for(int i = 0; i < ssp; ++i) {
			prgf.random_bool(tmp, length*bucket_size);
			X[party][i] = inProd(tmp, tKEYphi[party], length*bucket_size);
		}
		Hash::hash_once(dgst[party], X[party], sizeof(block)*ssp);

		for(int i = 1; i <= nP; ++i) for(int j = 1; j<= nP; ++j) if( (i < j) and (i == party or j == party) ) {
			int party2 = i + j - party;
			res.push_back(pool->enqueue([this, dgst, party2]() {
				io->send_data(party2, dgst[party], Hash::DIGEST_SIZE);
				io->recv_data(party2, dgst[party2], Hash::DIGEST_SIZE);
			}));
		}
		joinNclean(res);
		vector<future<bool>>	res2;

		for(int i = 1; i <= nP; ++i) for(int j = 1; j<= nP; ++j) if( (i < j) and (i == party or j == party) ) {
			int party2 = i + j - party;
			res2.push_back(pool->enqueue([this, X, dgst, party2]() -> bool {
				io->send_data(party2, X[party], sizeof(block)*ssp);
				io->recv_data(party2, X[party2], sizeof(block)*ssp);
				char tmp[Hash::DIGEST_SIZE];
				Hash::hash_once(tmp, X[party2], sizeof(block)*ssp);
				return strncmp(tmp, dgst[party2], Hash::DIGEST_SIZE)!=0;
			}));
		}
		if(joinNcleanCheat(res2)) error("commitment");

		for(int i = 2; i <= nP; ++i)
			xorBlocks_arr(X[1], X[1], X[i], ssp);
		for(int i = 0; i < ssp; ++i)X[2][i] = zero_block;
		if(!cmpBlock(X[1], X[2], ssp)) error("AND check");
	
		//land -> and	
		block S = sampleRandom<nP>(io, &prg, pool, party);

		int * ind = new int[length*bucket_size];
		int *location = new int[length*bucket_size];
		bool * d[nP+1];
		for(int i = 1; i <= nP; ++i)
			d[i] = new bool[length*(bucket_size-1)];
		for(int i = 0; i < length*bucket_size; ++i)
			location[i] = i;
		PRG prg2(&S);
		prg2.random_data(ind, length*bucket_size*4);
		for(int i = length*bucket_size-1; i>=0; --i) {
			int index = ind[i]%(i+1);
			index = index>0? index:(-1*index);
			int tmp = location[i];
			location[i] = location[index];
			location[index] = tmp;
		}
		delete[] ind;
	
		for(int i = 0; i < length; ++i) {
			for(int j = 0; j < bucket_size-1; ++j)
				d[party][(bucket_size-1)*i+j] = tr[3*location[i*bucket_size]+1] != tr[3*location[i*bucket_size+1+j]+1];
			for(int j = 1; j <= nP; ++j) if (j!= party) {
				memcpy(MAC[j]+3*i, tMAC[j]+3*location[i*bucket_size], 3*sizeof(block));
				memcpy(KEY[j]+3*i, tKEY[j]+3*location[i*bucket_size], 3*sizeof(block));
				for(int k = 1; k < bucket_size; ++k) {
					MAC[j][3*i] = MAC[j][3*i] ^ tMAC[j][3*location[i*bucket_size+k]];
					KEY[j][3*i] = KEY[j][3*i] ^ tKEY[j][3*location[i*bucket_size+k]];

					MAC[j][3*i+2] = MAC[j][3*i+2] ^ tMAC[j][3*location[i*bucket_size+k]+2];
					KEY[j][3*i+2] = KEY[j][3*i+2] ^ tKEY[j][3*location[i*bucket_size+k]+2];
				}
			}
			memcpy(r+3*i, tr+3*location[i*bucket_size], 3);
			for(int k = 1; k < bucket_size; ++k) {
				r[3*i] = r[3*i] != tr[3*location[i*bucket_size+k]];
				r[3*i+2] = r[3*i+2] != tr[3*location[i*bucket_size+k]+2];
			}
		}

		for(int i = 1; i <= nP; ++i) for(int j = 1; j<= nP; ++j) if( (i < j) and (i == party or j == party) ) {
			int party2 = i + j - party;
			res.push_back(pool->enqueue([this, d, length, bucket_size, party2]() {
				io->send_data(party2, d[party], (bucket_size-1)*length);
				io->flush(party2);
			}));
			res.push_back(pool->enqueue([this, d, length, bucket_size, party2]() {
				io->recv_data(party2, d[party2], (bucket_size-1)*length);
			}));
		}
		joinNclean(res);
		for(int i = 2; i <= nP; ++i)
			for(int j = 0; j <  (bucket_size-1)*length; ++j)
				d[1][j] = d[1][j]!=d[i][j];
		
		for(int i = 0; i < length; ++i)  {
			for(int j = 1; j <= nP; ++j)if (j!= party) { 
				for(int k = 1; k < bucket_size; ++k)
					if(d[1][(bucket_size-1)*i+k-1]) {
						MAC[j][3*i+2] = MAC[j][3*i+2] ^ tMAC[j][3*location[i*bucket_size+k]];
						KEY[j][3*i+2] = KEY[j][3*i+2] ^ tKEY[j][3*location[i*bucket_size+k]];
					}
			}
			for(int k = 1; k < bucket_size; ++k)
				if(d[1][(bucket_size-1)*i+k-1]) {
					r[3*i+2] = r[3*i+2] != tr[3*location[i*bucket_size+k]];
				}
		}
		
#ifdef __debug
		check_MAC(io, MAC, KEY, r, Delta, length*3, party);
		check_correctness(io, r, length, party);
#endif
	
//		ret.get();
		delete[] tr;
		delete[] phi;
		delete[] e;
		delete[] dgst;
		delete[] tmp;
		delete[] location;
		delete[] xs;
		for(int i = 1; i <= nP; ++i) {
			delete[] tMAC[i];
			delete[] tKEY[i];
			delete[] tMACphi[i];
			delete[] tKEYphi[i];
			delete[] X[i];
			delete[] s[i];
			delete[] d[i];
		}
		delete[] s[0];
	}

	//TODO: change to justGarble
	void garble(block * KEY, bool * r, bool * r2, int i, int I, int batch_size, uint8_t* data) {
		block tmp[GARB_BATCH_SIZE][4];
		block tmpH[GARB_BATCH_SIZE][4];
		// preprocess the inputs
		// as you normally would pre-process one
		// and accumulate them
		// mild care is needed to not mess up the indexing
		for (int ii = 0; ii < batch_size; ++ii) {
			tmp[ii][0] = KEY[3 * (ii+i)];
			tmp[ii][1] = tmp[ii][0] ^ Delta;
			tmp[ii][2] = KEY[3 * (ii + i) + 1];
			tmp[ii][3] = tmp[ii][2] ^ Delta;
		}

		// hand a big chunk of tasks off to cryptographic processing
		HnID(prps+I, (block*) tmp, (block*)tmp, 4*i, 4*batch_size, (block*)tmpH);

		// and then post-process as you would normally but with an added loop
		for (int ii = 0; ii < batch_size; ++ii) {
			block tmp2[4];
			tmp2[0] = tmp[ii][0] ^ tmp[ii][2];
			tmp2[1] = tmp[ii][1] ^ tmp[ii][2];
			tmp2[2] = tmp[ii][0] ^ tmp[ii][3];
			tmp2[3] = tmp[ii][1] ^ tmp[ii][3];

			uint8_t ldata = 0;
			ldata = LSB(tmp2[0]);
			ldata |= (LSB(tmp2[1]) << 1);
			ldata |= (LSB(tmp2[2]) << 2);
			ldata |= (LSB(tmp2[3]) << 3);
			if (((false != r[3 * (i+ii)]) && (false != r[3 * (i + ii) + 1])) != r2[(i + ii)])
				ldata = ldata ^ 0x1;
			if (((true != r[3 * (i + ii)]) && (false != r[3 * (i + ii) + 1])) != r2[(i + ii)])
				ldata = ldata ^ 0x2;
			if (((false != r[3 * (i + ii)]) && (true != r[3 * (i + ii) + 1])) != r2[(i + ii)])
				ldata = ldata ^ 0x4;
			if (((true != r[3 * (i + ii)]) && (true != r[3 * (i + ii) + 1])) != r2[(i + ii)])
				ldata = ldata ^ 0x8;
			data[ii] = ldata;
		}
	}

	// code changes analogous to garble() around the two prps invocations
	void evaluate(uint8_t tmp[EVAL_BATCH_SIZE], block * MAC, bool * r, int i, int I, int batch_size, bool outtmp[EVAL_BATCH_SIZE]) {
		block hin[EVAL_BATCH_SIZE], hin2[EVAL_BATCH_SIZE];
		for (int ii = 0; ii < batch_size; ++ii) {
			hin[ii] = sigma(MAC[3 * (i + ii)]) ^ makeBlock(0, 4 * (i+ii) + r[3 * (i + ii)]);
			hin2[ii] = sigma(MAC[3 * (i + ii) + 1]) ^ makeBlock(0, 4 * (i + ii) + 2 + r[3 * (i + ii) + 1]);
		}
		
		prps[I].H<EVAL_BATCH_SIZE>(hin, hin);
		prps[I].H<EVAL_BATCH_SIZE>(hin2, hin2);

		for (int ii = 0; ii < batch_size; ++ii) {
			block hB = hin[ii] ^ hin2[ii];
			uint8_t res = LSB(hB);
			uint8_t ltmp = tmp[ii];
			ltmp >>= (r[3 * (i + ii) + 1] * 2 + r[3 * (i + ii)]);
			outtmp[ii] = (ltmp & 0x1) != (res & 0x1);
		}
	}	

	void check_MAC_phi(block * MAC[nP+1], block * KEY[nP+1], block * phi, bool * r, int length) {
		block * tmp = new block[length];
		block *tD = new block[length];
		for(int i = 1; i <= nP; ++i) for(int j = 1; j <= nP; ++j) if (i < j) {
			if(party == i) {
				io->send_data(j, phi, length*sizeof(block));
				io->send_data(j, KEY[j], sizeof(block)*length);
				io->flush(j);
			} else if(party == j) {
				io->recv_data(i, tD, length*sizeof(block));
				io->recv_data(i, tmp, sizeof(block)*length);
				for(int k = 0; k < length; ++k) {
					if(r[k])tmp[k] = tmp[k] ^ tD[k];
				}
				if(!cmpBlock(MAC[i], tmp, length))
					error("check_MAC_phi failed!");
			}
		}
		delete[] tmp;
		delete[] tD;
		if(party == 1)
			cerr<<"check_MAC_phi pass!\n"<<flush; 
	}


	void check_zero(block * b, int l) {
		if(party == 1) {
			block * tmp1 = new block[l];
			block * tmp2 = new block[l];
			memcpy(tmp1, b, l*sizeof(block));
			for(int i = 2; i <= nP; ++i) {
				io->recv_data(i, tmp2, l*sizeof(block));
				xorBlocks_arr(tmp1, tmp1, tmp2, l);
			}
			block z = zero_block;
			for(int i = 0; i < l; ++i)
				if(!cmpBlock(&z, &tmp1[i], 1))
					error("check sum zero failed!");
			cerr<<"check zero sum pass!\n"<<flush; 
			delete[] tmp1;
			delete[] tmp2;
		} else {
			io->send_data(1, b, l*sizeof(block));
			io->flush(1);
		}
	}
	void HnID(CRH* crh, block*out, block* in, uint64_t id, int length, block * scratch = nullptr) {
		bool del = false;
		if(scratch == nullptr) {
			del = true;
			scratch = new block[length];
		}
		for(int i = 0; i < length; ++i){
			out[i] = scratch[i] = sigma(in[i]) ^ makeBlock(0, id);
			++id;
		}
		crh->permute_block(scratch, length);
		xorBlocks_arr(out, scratch, out, length);
		if(del) {
			delete[] scratch;
			scratch = nullptr;
		}
	}

	// code changes analogous to garble() around HnID
	void HnIDProcess(block* const tKEY[nP+1], block* const tKEYphi[nP + 1], block* phi, int total_iterations, int party2, int& k) {
		const int current_batch = std::min(total_iterations - k, HNID_BATCH_SIZE);
		block bH[HNID_BATCH_SIZE][2], tmpH[HNID_BATCH_SIZE][2];
		for (int kk = 0; kk < current_batch; ++kk) {
			bH[kk][0] = tKEY[party2][3 * (k + kk)];
			bH[kk][1] = bH[kk][0] ^ Delta;
		}

		HnID(prps + party2, (block*)bH, (block*)bH, 2 * k, 2* current_batch, (block*)tmpH);

		for (int kk = 0; kk < current_batch; ++kk) {
			tKEYphi[party2][k+kk] = bH[kk][0];
			bH[kk][1] = bH[kk][0] ^ bH[kk][1];
			bH[kk][1] = phi[k + kk] ^ bH[kk][1];
			io->send_data(party2, &bH[kk][1], sizeof(block));
		}
		k += current_batch;
	}

	// code changes analogous to garble() around prps2
	void RecvProcess(block* const tMAC[nP + 1], block* const tMACphi[nP + 1], bool* tr, int total_iterations, int party2, int& k) {
		const int current_batch = std::min(total_iterations - k, RECV_BATCH_SIZE);
		block bH[RECV_BATCH_SIZE], hin[RECV_BATCH_SIZE];
		io->recv_data(party2, bH, current_batch * sizeof(block));
		for (int kk = 0; kk < current_batch; ++kk) {
			hin[kk] = sigma(tMAC[party2][3 * (kk + k)]) ^ makeBlock(0, 2 * (kk + k) + tr[3 * (kk + k)]);
		}
		prps2[party2].H<RECV_BATCH_SIZE>(hin, hin);
		for(int kk = 0; kk < current_batch; ++kk) {
			tMACphi[party2][kk+k] = hin[kk];
			if (tr[3 * (kk+k)])tMACphi[party2][(kk + k)] = tMACphi[party2][(kk + k)] ^ bH[kk];
		}
		
		k += current_batch;
	}


};
#endif// FPRE_H__
