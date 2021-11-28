#ifndef EMP_AG2PC_AMORTIZED_C2PC_H__
#define EMP_AG2PC_AMORTIZED_C2PC_H__
#include <emp-tool/emp-tool.h>
#include <emp-ot/emp-ot.h>
#include "emp-ag2pc/fpre.h"

//#define __debug
namespace emp {
template<typename T, int exec>
class AmortizedC2PC {
public:
	static constexpr size_t GARBLING_BATCH_SIZE = 2;
	static constexpr size_t ONLINE_BATCH_SIZE = 8;
	const static int SSP = 5;//5*8 in fact...
	const block MASK = makeBlock(0x0ULL, 0xFFFFFULL);
	Fpre<T>* fpre = nullptr;
	block* mac[exec];
	block* key[exec];

	block* preprocess_mac;
	block* preprocess_key;

	block* sigma_mac[exec];
	block* sigma_key[exec];

	block* labels[exec];

	bool* mask[exec];
	BristolFormat* cf;
	T* io;
	int num_ands = 0;
	int party, total_pre;
	int exec_times = 0;
	bool* x1[exec];
	bool* y1[exec];
	bool* x2[exec];
	bool* y2[exec];

	AmortizedC2PC(T* io, int party, BristolFormat* cf) {
		this->party = party;
		this->io = io;
		this->cf = cf;
		for (int i = 0; i < cf->num_gate; ++i) {
			if (cf->gates[4 * i + 3] == AND_GATE)
				++num_ands;
		}
		total_pre = cf->n1 + cf->n2 + num_ands;
		fpre = new Fpre<T>(io, party, 5000000);

		preprocess_mac = new block[total_pre * exec];
		preprocess_key = new block[total_pre * exec];

		for (int i = 0; i < exec; ++i) {
			x1[i] = new bool[num_ands];
			y1[i] = new bool[num_ands];
			x2[i] = new bool[num_ands];
			y2[i] = new bool[num_ands];

			key[i] = new block[cf->num_wire];
			mac[i] = new block[cf->num_wire];

			//sigma values in the paper
			sigma_mac[i] = new block[num_ands];
			sigma_key[i] = new block[num_ands];

			labels[i] = new block[cf->num_wire];
			GT[i] = new block[num_ands][4][2];
			GTK[i] = new block[num_ands][4];
			GTM[i] = new block[num_ands][4];
			mask[i] = new bool[cf->n1 + cf->n2];
		}
	}
	~AmortizedC2PC() {
		for (int i = 0; i < exec; ++i) {
			delete[] key[i];
			delete[] mac[i];
			delete[] GT[i];
			delete[] GTK[i];
			delete[] GTM[i];

			delete[] sigma_mac[i];
			delete[] sigma_key[i];

			delete[] labels[i];
			delete[] mask[i];
			delete[] x1[i];
			delete[] x2[i];
			delete[] y1[i];
			delete[] y2[i];

		}
		delete[] preprocess_mac;
		delete[] preprocess_key;
		delete fpre;
	}

	PRG prg;
	PRP prp;
	block(*GT[exec])[4][2];
	block(*GTK[exec])[4];
	block(*GTM[exec])[4];

	//not allocation
	block* ANDS_mac[exec];
	block* ANDS_key[exec];
	void function_independent() {
		if (party == ALICE) {
			for (int e = 0; e < exec; ++e)
				prg.random_block(labels[e], total_pre);
		}

		fpre->refill();
		for (int e = 0; e < exec; ++e) {
			ANDS_mac[e] = fpre->MAC + 3 * e * num_ands;
			ANDS_key[e] = fpre->KEY + 3 * e * num_ands;
		}

		if (fpre->party == ALICE) {
			fpre->abit1[0]->send_dot(preprocess_key, exec * total_pre);
			fpre->io[0]->flush();
			fpre->abit2[0]->recv_dot(preprocess_mac, exec * total_pre);
			fpre->io2[0]->flush();
		}
		else {
			fpre->abit1[0]->recv_dot(preprocess_mac, exec * total_pre);
			fpre->io[0]->flush();
			fpre->abit2[0]->send_dot(preprocess_key, exec * total_pre);
			fpre->io2[0]->flush();
		}
		for (int i = 0; i < exec; ++i) {
			memcpy(key[i], preprocess_key + total_pre * i, (cf->n1 + cf->n2) * sizeof(block));
			memcpy(mac[i], preprocess_mac + total_pre * i, (cf->n1 + cf->n2) * sizeof(block));
		}
	}

	void function_dependent_st() {
		int ands = cf->n1 + cf->n2;

		for (int i = 0; i < cf->num_gate; ++i) {
			if (cf->gates[4 * i + 3] == AND_GATE) {
				for (int e = 0; e < exec; ++e) {
					key[e][cf->gates[4 * i + 2]] = preprocess_key[e * total_pre + ands];
					mac[e][cf->gates[4 * i + 2]] = preprocess_mac[e * total_pre + ands];
				}
				++ands;
			}
		}

		for (int e = 0; e < exec; ++e)
			for (int i = 0; i < cf->num_gate; ++i) {
				if (cf->gates[4 * i + 3] == XOR_GATE) {
					key[e][cf->gates[4 * i + 2]] = key[e][cf->gates[4 * i]] ^ key[e][cf->gates[4 * i + 1]];
					mac[e][cf->gates[4 * i + 2]] = mac[e][cf->gates[4 * i]] ^ mac[e][cf->gates[4 * i + 1]];
					if (party == ALICE)
						labels[e][cf->gates[4 * i + 2]] = labels[e][cf->gates[4 * i]] ^ labels[e][cf->gates[4 * i + 1]];
				}
				else if (cf->gates[4 * i + 3] == NOT_GATE) {
					if (party == ALICE)
						labels[e][cf->gates[4 * i + 2]] = labels[e][cf->gates[4 * i]] ^ fpre->Delta;

					key[e][cf->gates[4 * i + 2]] = key[e][cf->gates[4 * i]];
					mac[e][cf->gates[4 * i + 2]] = mac[e][cf->gates[4 * i]];
				}
			}
		for (int e = 0; e < exec; ++e) {
			ands = 0;
			for (int i = 0; i < cf->num_gate; ++i) {
				if (cf->gates[4 * i + 3] == AND_GATE) {
					x1[e][ands] = getLSB(mac[e][cf->gates[4 * i]] ^ ANDS_mac[e][3 * ands]);
					y1[e][ands] = getLSB(mac[e][cf->gates[4 * i + 1]] ^ ANDS_mac[e][3 * ands + 1]);
					ands++;
				}
			}
		}

		if (party == ALICE) {
			for (int e = 0; e < exec; ++e) {
				io->send_bool(x1[e], num_ands);
				io->send_bool(y1[e], num_ands);
			}
			for (int e = 0; e < exec; ++e) {
				io->recv_bool(x2[e], num_ands);
				io->recv_bool(y2[e], num_ands);
			}
		}
		else {
			for (int e = 0; e < exec; ++e) {
				io->recv_bool(x2[e], num_ands);
				io->recv_bool(y2[e], num_ands);
			}
			for (int e = 0; e < exec; ++e) {
				io->send_bool(x1[e], num_ands);
				io->send_bool(y1[e], num_ands);
			}
		}

		for (int e = 0; e < exec; ++e)
			for (int i = 0; i < num_ands; ++i) {
				x1[e][i] = logic_xor(x1[e][i], x2[e][i]);
				y1[e][i] = logic_xor(y1[e][i], y2[e][i]);
			}
		for (int e = 0; e < exec; ++e) {
			ands = 0;
			for (int i = 0; i < cf->num_gate; ++i) {
				if (cf->gates[4 * i + 3] == AND_GATE) {
					sigma_mac[e][ands] = ANDS_mac[e][3 * ands + 2];
					sigma_key[e][ands] = ANDS_key[e][3 * ands + 2];
					if (x1[e][ands]) {
						sigma_mac[e][ands] = sigma_mac[e][ands] ^ ANDS_mac[e][3 * ands + 1];
						sigma_key[e][ands] = sigma_key[e][ands] ^ ANDS_key[e][3 * ands + 1];
					}
					if (y1[e][ands]) {
						sigma_mac[e][ands] = sigma_mac[e][ands] ^ ANDS_mac[e][3 * ands];
						sigma_key[e][ands] = sigma_key[e][ands] ^ ANDS_key[e][3 * ands];
					}
					if (x1[e][ands] and y1[e][ands]) {
						if (party == ALICE) {
							sigma_key[e][ands] = sigma_key[e][ands] ^ fpre->Delta;
							sigma_key[e][ands] = sigma_key[e][ands] ^ makeBlock(0, 1);
						}
						else
							sigma_mac[e][ands] = sigma_mac[e][ands] ^ makeBlock(0, 1);
					}
#ifdef __debug
					block MM[] = { mac[e][cf->gates[4 * i]], mac[e][cf->gates[4 * i + 1]], sigma_mac[e][ands] };
					block KK[] = { key[e][cf->gates[4 * i]], key[e][cf->gates[4 * i + 1]], sigma_key[e][ands] };
					bool VV[] = { value[e][cf->gates[4 * i]], value[e][cf->gates[4 * i + 1]], sigma_value[e][ands] };
					check(MM, KK, VV);
#endif
					ands++;
				}
			}//sigma_[] stores the and of input wires to each AND gates
		}


		for (int e = 0; e < exec; ++e) {
			int indices[GARBLING_BATCH_SIZE];
			size_t collected_indices = 0;
			ands = 0;
			for (int i = 0; i < cf->num_gate; ++i) {
				if (cf->gates[4 * i + 3] == AND_GATE) {
					indices[collected_indices] = i;
					collected_indices++;
				}
				if (collected_indices == GARBLING_BATCH_SIZE) {
					GarblingHashing(indices, ands, collected_indices, e, io);
					collected_indices = 0;
				}
			}
			GarblingHashing(indices, ands, collected_indices, e, io);
		}

		block tmp;
		if (party == ALICE) {
			for (int e = 0; e < exec; ++e)
				send_partial_block<T, SSP>(io, mac[e], cf->n1);

			for (int e = 0; e < exec; ++e)
				for (int i = cf->n1; i < cf->n1 + cf->n2; ++i) {
					recv_partial_block<T, SSP>(io, &tmp, 1);
					block ttt = key[e][i] ^ fpre->Delta;
					ttt = ttt & MASK;
					block mask_key = key[e][i] & MASK;
					tmp = tmp & MASK;

					if (cmpBlock(&tmp, &mask_key, 1))
						mask[e][i] = false;
					else if (cmpBlock(&tmp, &ttt, 1))
						mask[e][i] = true;
					else cout << "no match! ALICE\t" << i << endl;
				}
		}
		else {
			for (int e = 0; e < exec; ++e)
				for (int i = 0; i < cf->n1; ++i) {
					recv_partial_block<T, SSP>(io, &tmp, 1);
					block ttt = key[e][i] ^ fpre->Delta;
					ttt = ttt & MASK;
					tmp = tmp & MASK;
					block mask_key = key[e][i] & MASK;

					if (cmpBlock(&tmp, &mask_key, 1)) {
						mask[e][i] = false;
					}
					else if (cmpBlock(&tmp, &ttt, 1)) {
						mask[e][i] = true;
					}
					else cout << "no match! BOB\t" << i << endl;
				}
			for (int e = 0; e < exec; ++e)
				send_partial_block<T, SSP>(io, mac[e] + cf->n1, cf->n2);
		}
	}

	void function_dependent() {
		int e = 0;
		vector<future<void>> res;
		for (int I = 0; I < fpre->THDS and e < exec; ++I, ++e) {
			res.push_back(fpre->pool->enqueue(
				&AmortizedC2PC::function_dependent_thread, this, e, I
			));
		}
		while (e < exec) {
			for (int I = 0; I < fpre->THDS and e < exec; ++I, ++e) {
				res[I].get();
				res[I] = fpre->pool->enqueue(
					&AmortizedC2PC::function_dependent_thread, this, e, I);
			}
		}
		for (auto& a : res)
			a.get();
	}
	void function_dependent_thread(int e, int I) {
		int ands = cf->n1 + cf->n2;

		for (int i = 0; i < cf->num_gate; ++i) {
			if (cf->gates[4 * i + 3] == AND_GATE) {
				key[e][cf->gates[4 * i + 2]] = preprocess_key[e * total_pre + ands];
				mac[e][cf->gates[4 * i + 2]] = preprocess_mac[e * total_pre + ands];
				++ands;
			}
		}

		for (int i = 0; i < cf->num_gate; ++i) {
			if (cf->gates[4 * i + 3] == XOR_GATE) {
				key[e][cf->gates[4 * i + 2]] = key[e][cf->gates[4 * i]] ^ key[e][cf->gates[4 * i + 1]];
				mac[e][cf->gates[4 * i + 2]] = mac[e][cf->gates[4 * i]] ^ mac[e][cf->gates[4 * i + 1]];
				if (party == ALICE)
					labels[e][cf->gates[4 * i + 2]] = labels[e][cf->gates[4 * i]] ^ labels[e][cf->gates[4 * i + 1]];;
			}
			else if (cf->gates[4 * i + 3] == NOT_GATE) {
				if (party == ALICE)
					labels[e][cf->gates[4 * i + 2]] = labels[e][cf->gates[4 * i]] ^ fpre->Delta;
				key[e][cf->gates[4 * i + 2]] = key[e][cf->gates[4 * i]];
				mac[e][cf->gates[4 * i + 2]] = mac[e][cf->gates[4 * i]];
			}
		}
		ands = 0;
		for (int i = 0; i < cf->num_gate; ++i) {
			if (cf->gates[4 * i + 3] == AND_GATE) {
				x1[e][ands] = getLSB(mac[e][cf->gates[4 * i]] ^ ANDS_mac[e][3 * ands]);
				y1[e][ands] = getLSB(mac[e][cf->gates[4 * i + 1]] ^ ANDS_mac[e][3 * ands + 1]);
				ands++;
			}
		}

		if (party == ALICE) {
			fpre->io2[I]->send_bool(x1[e], num_ands);
			fpre->io2[I]->send_bool(y1[e], num_ands);
			fpre->io2[I]->recv_bool(x2[e], num_ands);
			fpre->io2[I]->recv_bool(y2[e], num_ands);
		}
		else {
			fpre->io2[I]->recv_bool(x2[e], num_ands);
			fpre->io2[I]->recv_bool(y2[e], num_ands);
			fpre->io2[I]->send_bool(x1[e], num_ands);
			fpre->io2[I]->send_bool(y1[e], num_ands);
		}

		for (int i = 0; i < num_ands; ++i) {
			x1[e][i] = logic_xor(x1[e][i], x2[e][i]);
			y1[e][i] = logic_xor(y1[e][i], y2[e][i]);
		}
		ands = 0;
		for (int i = 0; i < cf->num_gate; ++i) {
			if (cf->gates[4 * i + 3] == AND_GATE) {
				sigma_mac[e][ands] = ANDS_mac[e][3 * ands + 2];
				sigma_key[e][ands] = ANDS_key[e][3 * ands + 2];
				if (x1[e][ands]) {
					sigma_mac[e][ands] = sigma_mac[e][ands] ^ ANDS_mac[e][3 * ands + 1];
					sigma_key[e][ands] = sigma_key[e][ands] ^ ANDS_key[e][3 * ands + 1];
				}
				if (y1[e][ands]) {
					sigma_mac[e][ands] = sigma_mac[e][ands] ^ ANDS_mac[e][3 * ands];
					sigma_key[e][ands] = sigma_key[e][ands] ^ ANDS_key[e][3 * ands];
				}
				if (x1[e][ands] and y1[e][ands]) {
					if (party == ALICE) {
						sigma_key[e][ands] = sigma_key[e][ands] ^ fpre->Delta;
						sigma_key[e][ands] = sigma_key[e][ands] ^ makeBlock(0, 1);

					}
					else
						sigma_mac[e][ands] = sigma_mac[e][ands] ^ makeBlock(0, 1);
				}
#ifdef __debug
				block MM[] = { mac[e][cf->gates[4 * i]], mac[e][cf->gates[4 * i + 1]], sigma_mac[e][ands] };
				block KK[] = { key[e][cf->gates[4 * i]], key[e][cf->gates[4 * i + 1]], sigma_key[e][ands] };
				bool VV[] = { value[e][cf->gates[4 * i]], value[e][cf->gates[4 * i + 1]], sigma_value[e][ands] };
				check(MM, KK, VV);
#endif
				ands++;
			}
		}//sigma_[] stores the and of input wires to each AND gates

		//block H[4][2];
		//block K[4], M[4];
		ands = 0;
		int indices[GARBLING_BATCH_SIZE];
		size_t collected_indices = 0;
		for (int i = 0; i < cf->num_gate; ++i) {
			if (cf->gates[4 * i + 3] == AND_GATE) {
				indices[collected_indices] = i;
				collected_indices++;
			}
			if (collected_indices == GARBLING_BATCH_SIZE) {
				GarblingHashing(indices, ands, collected_indices, e, fpre->io2[I]);
				collected_indices = 0;
			}
		}
		GarblingHashing(indices, ands, collected_indices, e, fpre->io2[I]);

		block tmp;
		if (party == ALICE) {
			send_partial_block<T, SSP>(fpre->io2[I], mac[e], cf->n1);
			for (int i = cf->n1; i < cf->n1 + cf->n2; ++i) {
				recv_partial_block<T, SSP>(fpre->io2[I], &tmp, 1);
				block ttt = key[e][i] ^ fpre->Delta;
				ttt = ttt & MASK;
				block mask_key = key[e][i] & MASK;
				tmp = tmp & MASK;
				if (cmpBlock(&tmp, &mask_key, 1))
					mask[e][i] = false;
				else if (cmpBlock(&tmp, &ttt, 1))
					mask[e][i] = true;
				else cout << "no match! ALICE\t" << i << endl;
			}
		}
		else {
			for (int i = 0; i < cf->n1; ++i) {
				recv_partial_block<T, SSP>(fpre->io2[I], &tmp, 1);
				block ttt = key[e][i] ^ fpre->Delta;
				ttt = ttt & MASK;
				tmp = tmp & MASK;
				block mask_key = key[e][i] & MASK;
				if (cmpBlock(&tmp, &mask_key, 1)) {
					mask[e][i] = false;
				}
				else if (cmpBlock(&tmp, &ttt, 1)) {
					mask[e][i] = true;
				}
				else cout << "no match! BOB\t" << i << endl;
			}
			send_partial_block<T, SSP>(fpre->io2[I], mac[e] + cf->n1, cf->n2);
		}
		fpre->io2[I]->flush();
	}


	void online(bool* input, bool* output) {
		uint8_t* mask_input = new uint8_t[cf->num_wire];
		memset(mask_input, 0, cf->num_wire);
		block tmp;
#ifdef __debug
		for (int e = 0; e < exec; ++e)
			for (int i = 0; i < cf->n1 + cf->n2; ++i)
				check2(mac[e][i], key[e][i], value[e][i]);
#endif
		if (party == ALICE) {
			for (int i = cf->n1; i < cf->n1 + cf->n2; ++i) {
				mask_input[i] = logic_xor(input[i - cf->n1], getLSB(mac[exec_times][i]));
				mask_input[i] = logic_xor(mask_input[i], mask[exec_times][i]);
			}
			io->recv_data(mask_input, cf->n1);
			io->send_data(mask_input + cf->n1, cf->n2);
			for (int i = 0; i < cf->n1 + cf->n2; ++i) {
				tmp = labels[exec_times][i];
				if (mask_input[i]) tmp = tmp ^ fpre->Delta;
				io->send_block(&tmp, 1);
			}
			//send output mask data
			send_partial_block<T, SSP>(io, mac[exec_times] + cf->num_wire - cf->n3, cf->n3);
		}
		else {
			for (int i = 0; i < cf->n1; ++i) {
				mask_input[i] = logic_xor(input[i], getLSB(mac[exec_times][i]));
				mask_input[i] = logic_xor(mask_input[i], mask[exec_times][i]);
			}
			io->send_data(mask_input, cf->n1);
			io->recv_data(mask_input + cf->n1, cf->n2);
			io->recv_block(labels[exec_times], cf->n1 + cf->n2);
		}
		int ands = 0;
		if (party == BOB) {
			int queued_indices[ONLINE_BATCH_SIZE];
			int queued_gate_ids[ONLINE_BATCH_SIZE];
			size_t queue_size = 0;
			for (int i = 0; i < cf->num_gate; ++i) {
				bool trap = queue_size == ONLINE_BATCH_SIZE;
				int left_in = cf->gates[4 * i];
				for (size_t t = 0; t < queue_size; ++t)
					if (left_in == queued_gate_ids[t])
						trap = true;
				if (cf->gates[4 * i + 3] == XOR_GATE || cf->gates[4 * i + 3] == AND_GATE) {
					int right_in = cf->gates[4 * i+1];
					for (size_t t = 0; t < queue_size; ++t)
						if (right_in == queued_gate_ids[t])
							trap = true;
				}
				if (trap) {
					EvaluateANDGates(mask_input, queued_indices,queue_size,ands);
					queue_size = 0;
				}

				if (cf->gates[4 * i + 3] == XOR_GATE) {
					labels[exec_times][cf->gates[4 * i + 2]] = labels[exec_times][cf->gates[4 * i]] ^ labels[exec_times][cf->gates[4 * i + 1]];
					mask_input[cf->gates[4 * i + 2]] = logic_xor(mask_input[cf->gates[4 * i]], mask_input[cf->gates[4 * i + 1]]);
				}
				else if (cf->gates[4 * i + 3] == AND_GATE) {
					queued_indices[queue_size] = i;
					queued_gate_ids[queue_size] = cf->gates[4 * i + 2];
					queue_size++;
				}
				else {
					mask_input[cf->gates[4 * i + 2]] = not mask_input[cf->gates[4 * i]];
					labels[exec_times][cf->gates[4 * i + 2]] = labels[exec_times][cf->gates[4 * i]];
				}
			}
			EvaluateANDGates(mask_input, queued_indices, queue_size, ands);
		}
		if (party == BOB) {
			bool* o = new bool[cf->n3];
			for (int i = 0; i < cf->n3; ++i) {
				block tmp;
				recv_partial_block<T, SSP>(io, &tmp, 1);
				tmp = tmp & MASK;

				block ttt = key[exec_times][cf->num_wire - cf->n3 + i] ^ fpre->Delta;
				ttt = ttt & MASK;
				key[exec_times][cf->num_wire - cf->n3 + i] = key[exec_times][cf->num_wire - cf->n3 + i] & MASK;

				if (cmpBlock(&tmp, &key[exec_times][cf->num_wire - cf->n3 + i], 1))
					o[i] = false;
				else if (cmpBlock(&tmp, &ttt, 1))
					o[i] = true;
				else 	cout << "no match output label!" << endl;
			}
			for (int i = 0; i < cf->n3; ++i) {
				output[i] = logic_xor(o[i], mask_input[cf->num_wire - cf->n3 + i]);
				output[i] = logic_xor(output[i], getLSB(mac[exec_times][cf->num_wire - cf->n3 + i]));
			}
			delete[] o;
		}
		exec_times++;
		delete[] mask_input;
	}

	__attribute__((always_inline))
	void Hash(block H[GARBLING_BATCH_SIZE][4][2], const block a[GARBLING_BATCH_SIZE], const block b[GARBLING_BATCH_SIZE], int i[GARBLING_BATCH_SIZE], size_t num_gates) {
		for (size_t bidx = 0; bidx < num_gates; ++bidx) {
			block A[2], B[2];
			A[0] = a[bidx]; A[1] = a[bidx] ^ fpre->Delta;
			B[0] = b[bidx]; B[1] = b[bidx] ^ fpre->Delta;
			A[0] = sigma(A[0]);
			A[1] = sigma(A[1]);
			B[0] = sigma(sigma(B[0]));
			B[1] = sigma(sigma(B[1]));

			H[bidx][0][1] = H[bidx][0][0] = A[0] ^ B[0];
			H[bidx][1][1] = H[bidx][1][0] = A[0] ^ B[1];
			H[bidx][2][1] = H[bidx][2][0] = A[1] ^ B[0];
			H[bidx][3][1] = H[bidx][3][0] = A[1] ^ B[1];
			for (uint64_t j = 0; j < 4; ++j) {
				H[bidx][j][0] = H[bidx][j][0] ^ _mm_set_epi64x(4 * static_cast<uint64_t>(i[bidx]) + j, 0ULL);
				H[bidx][j][1] = H[bidx][j][1] ^ _mm_set_epi64x(4 * static_cast<uint64_t>(i[bidx]) + j, 1ULL);
			}
		}
		prp.permute_block((block *)H, 8* num_gates);
	}

	__attribute__((always_inline))
	void Hash(block H[ONLINE_BATCH_SIZE][2], block a[ONLINE_BATCH_SIZE], block b[ONLINE_BATCH_SIZE], int i[ONLINE_BATCH_SIZE], int row[ONLINE_BATCH_SIZE], size_t num_entries) {
		for (size_t ii = 0; ii < num_entries; ++ii) {
			a[ii] = sigma(a[ii]);
			b[ii] = sigma(sigma(b[ii]));
			H[ii][0] = H[ii][1] = a[ii] ^ b[ii];
			H[ii][0] = H[ii][0] ^ _mm_set_epi64x(4 * static_cast<uint64_t>(i[ii]) + static_cast<uint64_t>(row[ii]), 0ULL);
			H[ii][1] = H[ii][1] ^ _mm_set_epi64x(4 * static_cast<uint64_t>(i[ii]) + static_cast<uint64_t>(row[ii]), 1ULL);
		}
		prp.permute_block((block *)H, 2*num_entries);
	}


	bool logic_xor(bool a, bool b) {
		return a!= b;
		if(a) if(not b) return true;
		if(not a) if(b) return true;
		return false;
	}
	string tostring(bool a) {
		if(a) return "T";
		else return "F";
	}
private:
	__attribute__((always_inline))
	void GarblingHashing(int indices[GARBLING_BATCH_SIZE], int& ands, size_t num_gates, const int e, T* io) {
		block H[GARBLING_BATCH_SIZE][4][2];
		block K[GARBLING_BATCH_SIZE][4], M[GARBLING_BATCH_SIZE][4];
		block left[GARBLING_BATCH_SIZE], right[GARBLING_BATCH_SIZE];

		int input_ands = ands;

		for (size_t ii = 0; ii < num_gates; ++ii) {
			int i = indices[ii];
			left[ii] = labels[e][cf->gates[4 * i]];
			right[ii] = labels[e][cf->gates[4 * i + 1]];
			M[ii][0] = sigma_mac[e][ands] ^ mac[e][cf->gates[4 * i + 2]];
			M[ii][1] = M[ii][0] ^ mac[e][cf->gates[4 * i]];
			M[ii][2] = M[ii][0] ^ mac[e][cf->gates[4 * i + 1]];
			M[ii][3] = M[ii][1] ^ mac[e][cf->gates[4 * i + 1]];
			if (party == BOB)
				M[ii][3] = M[ii][3] ^ makeBlock(0, 1);

			K[ii][0] = sigma_key[e][ands] ^ key[e][cf->gates[4 * i + 2]];
			K[ii][1] = K[ii][0] ^ key[e][cf->gates[4 * i]];
			K[ii][2] = K[ii][0] ^ key[e][cf->gates[4 * i + 1]];
			K[ii][3] = K[ii][1] ^ key[e][cf->gates[4 * i + 1]];
			if (party == ALICE) {
				K[ii][3] = K[ii][3] ^ fpre->Delta;
				K[ii][3] = K[ii][3] ^ makeBlock(0, 1);
			}
			++ands;
		}
		if (party == ALICE) {
			Hash(H, left, right, indices, num_gates);
		}
		ands = input_ands;
		for (size_t ii = 0; ii < num_gates; ++ii) {
			int i = indices[ii];
			if (party == ALICE) {
				for (int j = 0; j < 4; ++j) {
					H[ii][j][0] = H[ii][j][0] ^ M[ii][j];
					H[ii][j][1] = H[ii][j][1] ^ K[ii][j] ^ labels[e][cf->gates[4 * i + 2]];
					if (getLSB(M[ii][j]))
						H[ii][j][1] = H[ii][j][1] ^ fpre->Delta;
				}
				for (int j = 0; j < 4; ++j) {
					send_partial_block<T, SSP>(io, &H[ii][j][0], 1);
					io->send_block(&H[ii][j][1], 1);
				}
			}
			else {
				memcpy(GTK[e][ands], K[ii], sizeof(block) * 4);
				memcpy(GTM[e][ands], M[ii], sizeof(block) * 4);

				for (int j = 0; j < 4; ++j) {
					recv_partial_block<T, SSP>(io, &GT[e][ands][j][0], 1);
					io->recv_block(&GT[e][ands][j][1], 1);
				}
			}
			++ands;
		}
	}

	__attribute__((always_inline))
	void EvaluateANDGates(uint8_t* mask_input,int indices[ONLINE_BATCH_SIZE], size_t num_gates, int& ands) {
		int mask_indices[ONLINE_BATCH_SIZE];
		block lefts[ONLINE_BATCH_SIZE], rights[ONLINE_BATCH_SIZE];
		block H[ONLINE_BATCH_SIZE][2];
		for (size_t ii = 0; ii < num_gates; ++ii) {
			int i = indices[ii];
			int index = 2 * mask_input[cf->gates[4 * i]] + mask_input[cf->gates[4 * i + 1]];
			mask_indices[ii] = index;
			lefts[ii] = labels[exec_times][cf->gates[4 * i]];
			rights[ii] = labels[exec_times][cf->gates[4 * i + 1]];
		}
		Hash(H, lefts, rights, indices, mask_indices, num_gates);
		for (size_t ii = 0; ii < num_gates; ++ii) {
			int i = indices[ii];
			int index = 2 * mask_input[cf->gates[4 * i]] + mask_input[cf->gates[4 * i + 1]];
			/*block H[2];
			Hash(H, labels[exec_times][cf->gates[4 * i]], labels[exec_times][cf->gates[4 * i + 1]], i, index);*/
			GT[exec_times][ands][index][0] = GT[exec_times][ands][index][0] ^ H[ii][0];
			GT[exec_times][ands][index][1] = GT[exec_times][ands][index][1] ^ H[ii][1];

			block ttt = GTK[exec_times][ands][index] ^ fpre->Delta;
			ttt = ttt & MASK;
			GTK[exec_times][ands][index] = GTK[exec_times][ands][index] & MASK;
			GT[exec_times][ands][index][0] = GT[exec_times][ands][index][0] & MASK;

			if (cmpBlock(&GT[exec_times][ands][index][0], &GTK[exec_times][ands][index], 1))
				mask_input[cf->gates[4 * i + 2]] = false;
			else if (cmpBlock(&GT[exec_times][ands][index][0], &ttt, 1))
				mask_input[cf->gates[4 * i + 2]] = true;
			else 	cout << ands << "no match GT!" << endl;
			mask_input[cf->gates[4 * i + 2]] = logic_xor(mask_input[cf->gates[4 * i + 2]], getLSB(GTM[exec_times][ands][index]));

			labels[exec_times][cf->gates[4 * i + 2]] = GT[exec_times][ands][index][1] ^ GTM[exec_times][ands][index];
			ands++;
		}
	}
};
}
#endif// AMORTIZED_C2PC_H__
