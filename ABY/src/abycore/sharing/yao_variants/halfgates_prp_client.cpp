#include "halfgates_prp_client.h"
#include "../aes_processors/aesni_halfgate_processors.h"
#include "../aes_processors/vaes_halfgate_processors.h"
#include "../cpu_features/include/cpuinfo_x86.h"

#include <emmintrin.h>

static const cpu_features::X86Features CPU_FEATURES = cpu_features::GetX86Info().features;

bool HalfGatesPRPClientSharing::evaluateXORGate(GATE* gate)
{
	uint32_t nvals = gate->nvals;
	uint32_t idleft = gate->ingates.inputs.twin.left; //gate->gs.ginput.left;
	uint32_t idright = gate->ingates.inputs.twin.right; //gate->gs.ginput.right;

	InstantiateGate(gate);
	//TODO: optimize for uint64_t pointers, there might be some problems here, code is untested
	/*for(uint32_t i = 0; i < m_nSecParamBytes * nvals; i++) {
	 gate->gs.yval[i] = m_vGates[idleft].gs.yval[i] ^ m_vGates[idright].gs.yval[i];
	 }*/
	 //std::cout << "doing " << m_nSecParamIters << "iters on " << nvals << " vals " << std::endl;
	// < m_nSecParamIters * nvals

	assert(m_nSecParamBytes == 16);

	for (uint32_t i = 0; i < nvals; i++) {
		__m128i leftKey = _mm_loadu_si128((__m128i*)(m_vGates[idleft].gs.yval + i * 16));
		__m128i rightKey = _mm_loadu_si128((__m128i*)(m_vGates[idright].gs.yval + i * 16));
		__m128i result = _mm_xor_si128(leftKey, rightKey);
		_mm_storeu_si128((__m128i*)(gate->gs.yval + i * 16), result);


		//((UGATE_T*)gate->gs.yval)[i] = ((UGATE_T*)m_vGates[idleft].gs.yval)[i] ^ ((UGATE_T*)m_vGates[idright].gs.yval)[i];
	}
	//std::cout << "Keyval (" << 0 << ")= " << (gate->gs.yval[m_nSecParamBytes-1] & 0x01)  << std::endl;
	//std::cout << (gate->gs.yval[m_nSecParamBytes-1] & 0x01);
#ifdef DEBUGYAOCLIENT
	PrintKey(gate->gs.yval);
	std::cout << " = ";
	PrintKey(m_vGates[idleft].gs.yval);
	std::cout << " (" << idleft << ") ^ ";
	PrintKey(m_vGates[idright].gs.yval);
	std::cout << " (" << idright << ")" << std::endl;
#endif

	UsedGate(idleft);
	UsedGate(idright);

	return true;
}

std::unique_ptr<AESProcessor> HalfGatesPRPClientSharing::provideEvaluationProcessor() const
{
	if (m_nSecParamBytes != 16)
	{
		std::cerr << "unsupported security parameter." << std::endl;
		assert(false);
	}
	else
	{
		if (ENABLE_VAES && CPU_FEATURES.vaes && CPU_FEATURES.avx512f && CPU_FEATURES.avx512bw && CPU_FEATURES.avx512vl) {
			if (ENABLE_HYBRID) {
				return std::make_unique<HybridHalfgateEvaluatingProcessor<FixedKeyLTEvaluatingVaesProcessor, FixedKeyLTEvaluatingAesniProcessor>>(getAndQueue(), m_vGates);
			}
			else {
				return std::make_unique<FixedKeyLTEvaluatingVaesProcessor>(getAndQueue(), m_vGates);
			}
		}	
		else if (CPU_FEATURES.aes && CPU_FEATURES.sse4_1)
			return std::make_unique<FixedKeyLTEvaluatingAesniProcessor>(getAndQueue(), m_vGates);
		else
		{
			std::cerr << "unsupported host CPU." << std::endl;
			assert(false);
		}
	}
}

void HalfGatesPRPClientSharing::evaluateDeferredANDGates(size_t numWires)
{
	// the buffers are needed for the batch processing
	for (auto* currentGate : getAndQueue())
		InstantiateGate(currentGate);

	// we call into another class here as this allows us to exploit dynamic dispatch
	// to switch between AES256, AES128, AES-NI and VAES as needed based on a ctor parameter
	// This essentially does the wire encrypting previously done just-in-time
	// TODO: If time, do some fancy software pipelining tricks and also do the actual table generation?
	m_aesProcessor->computeAESOutKeys(m_nAndGateTableCtr, numWires, m_vAndGateTable.GetArr());

	m_nAndGateTableCtr += numWires;

	for (auto* currentGate : getAndQueue())
	{
		uint32_t idleft = currentGate->ingates.inputs.twin.left; //gate->gs.ginput.left;
		uint32_t idright = currentGate->ingates.inputs.twin.right; //gate->gs.ginput.right;

		UsedGate(idleft);
		UsedGate(idright);
	}
}

// this is just a post-processing call
void HalfGatesPRPClientSharing::EvaluateGarbledTablePrepared(GATE* gate, uint32_t pos, GATE* gleft, GATE* gright)
{
	uint8_t* lkey, * rkey, * okey, * gtptr;
	uint8_t lpbit, rpbit;

	okey = gate->gs.yval + pos * m_nSecParamBytes;
	lkey = gleft->gs.yval + pos * m_nSecParamBytes;
	rkey = gright->gs.yval + pos * m_nSecParamBytes;
	gtptr = m_vAndGateTable.GetArr() + m_nSecParamBytes * KEYS_PER_GATE_IN_TABLE * m_nAndGateTableCtr;

	lpbit = lkey[m_nSecParamBytes - 1] & 0x01;
	rpbit = rkey[m_nSecParamBytes - 1] & 0x01;

	assert(lpbit < 2 && rpbit < 2);

	//EncryptWire(m_vTmpEncBuf[0], lkey, KEYS_PER_GATE_IN_TABLE*m_nAndGateTableCtr);
	//EncryptWire(m_vTmpEncBuf[1], rkey, KEYS_PER_GATE_IN_TABLE*m_nAndGateTableCtr+1);

	//m_pKeyOps->XOR(okey, m_vTmpEncBuf[0], m_vTmpEncBuf[1]);//gc_xor(okey, encbuf[0], encbuf[1]);

	if (lpbit) {
		m_pKeyOps->XOR(okey, okey, gtptr);//gc_xor(okey, okey, gtptr);
	}
	if (rpbit) {
		m_pKeyOps->XOR(okey, okey, gtptr + m_nSecParamBytes);//gc_xor(okey, okey, gtptr+BYTES_SSP);
		m_pKeyOps->XOR(okey, okey, lkey);//gc_xor(okey, okey, gtptr+BYTES_SSP);
	}

#ifdef DEBUGYAOCLIENT
	std::cout << " using: ";
	PrintKey(lkey);
	std::cout << " (" << (uint32_t)lpbit << ") and : ";
	PrintKey(rkey);
	std::cout << " (" << (uint32_t)rpbit << ") to : ";
	PrintKey(okey);
	std::cout << " (" << (uint32_t)(okey[m_nSecParamBytes - 1] & 0x01) << ")" << std::endl;
	std::cout << "A: ";
	PrintKey(m_vTmpEncBuf[0]);
	std::cout << "; B: ";
	PrintKey(m_vTmpEncBuf[1]);
	std::cout << std::endl;
	std::cout << "Table A: ";
	PrintKey(gtptr);
	std::cout << "; Table B: ";
	PrintKey(gtptr + m_nSecParamBytes);
	std::cout << std::endl;
#endif
}

bool HalfGatesPRPClientSharing::evaluateUNIVGate(GATE* gate) {
	uint32_t idleft = gate->ingates.inputs.twin.left; //gate->gs.ginput.left;
	uint32_t idright = gate->ingates.inputs.twin.right; //gate->gs.ginput.right;
	GATE* gleft = &(m_vGates[idleft]);
	GATE* gright = &(m_vGates[idright]);

	//evaluate univeral gate table
	InstantiateGate(gate);
	for (uint32_t g = 0; g < gate->nvals; g++) {
		EvaluateUniversalGate(gate, g, gleft, gright);
		m_nUniversalGateTableCtr++;
	}
	UsedGate(idleft);
	UsedGate(idright);

	return true;
}

void HalfGatesPRPClientSharing::EvaluateUniversalGate(GATE* gate, uint32_t pos, GATE* gleft, GATE* gright)
{
	BYTE* lkey, * rkey, * okey;
	uint32_t id;
	lkey = gleft->gs.yval + pos * m_nSecParamBytes;
	rkey = gright->gs.yval + pos * m_nSecParamBytes;
	okey = gate->gs.yval + pos * m_nSecParamBytes;

	id = (lkey[m_nSecParamBytes - 1] & 0x01) << 1;
	id += (rkey[m_nSecParamBytes - 1] & 0x01);

	//encrypt_wire((BYTE*)gate->gs.val, m_vGarbledTables.GetArr() + BYTES_SSP * (4 * andctr + id), pleft, pright, id, m_kGarble, key_buf);
	if (id == 0) {
		EncryptWireGRR3(okey, m_bZeroBuf, lkey, rkey, id);
#ifdef DEBUGYAOCLIENT
		std::cout << " decrypted : ";
		PrintKey(m_bZeroBuf);
#endif
	}
	else {
#ifdef DEBUGYAOCLIENT
		std::cout << " decrypted : ";
		PrintKey(m_vUniversalGateTable.GetArr() + m_nSecParamBytes * (KEYS_PER_UNIV_GATE_IN_TABLE * m_nUniversalGateTableCtr + id - 1));
#endif
		EncryptWireGRR3(okey, m_vUniversalGateTable.GetArr() + m_nSecParamBytes * (KEYS_PER_UNIV_GATE_IN_TABLE * m_nUniversalGateTableCtr + id - 1), lkey, rkey, id);
	}

#ifdef DEBUGYAOCLIENT
	std::cout << " using: ";
	PrintKey(lkey);
	std::cout << " and : ";
	PrintKey(rkey);
	std::cout << " to : ";
	PrintKey(okey);
	std::cout << std::endl;
#endif
}
