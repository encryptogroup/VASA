#include "prf_client.h"
#include "../aes_processors/aesni_prf_processors.h"
#include "../aes_processors/vaes_prf_processors.h"
#include "../cpu_features/include/cpuinfo_x86.h"

static const cpu_features::X86Features CPU_FEATURES = cpu_features::GetX86Info().features;

void PRFClientSharing::evaluateDeferredXORGates(size_t numWires)
{
	// the buffers are needed for the batch processing
	for (auto* currentGate : getXorQueue())
		InstantiateGate(currentGate);

	// we call into another class here as this allows us to exploit dynamic dispatch
	// to switch between AES256, AES128, AES-NI and VAES as needed based on a ctor parameter
	// This essentially does the wire encrypting previously done just-in-time
	m_xorAESProcessor->computeAESOutKeys(m_nXorGateTableCtr, numWires, m_vXorGateTable.GetArr());

	m_nXorGateTableCtr += numWires;

	for (auto* currentGate : getXorQueue())
	{
		uint32_t idleft = currentGate->ingates.inputs.twin.left; //gate->gs.ginput.left;
		uint32_t idright = currentGate->ingates.inputs.twin.right; //gate->gs.ginput.right;

		UsedGate(idleft);
		UsedGate(idright);
	}
}

void PRFClientSharing::evaluateDeferredANDGates(size_t numWires)
{
	// the buffers are needed for the batch processing
	for (auto* currentGate : getAndQueue())
		InstantiateGate(currentGate);

	// we call into another class here as this allows us to exploit dynamic dispatch
	// to switch between AES256, AES128, AES-NI and VAES as needed based on a ctor parameter
	// This essentially does the wire encrypting previously done just-in-time
	m_andAESProcessor->computeAESOutKeys(m_nAndGateTableCtr, numWires, m_vAndGateTable.GetArr());

	m_nAndGateTableCtr += numWires;

	for (auto* currentGate : getAndQueue())
	{
		uint32_t idleft = currentGate->ingates.inputs.twin.left; //gate->gs.ginput.left;
		uint32_t idright = currentGate->ingates.inputs.twin.right; //gate->gs.ginput.right;

		UsedGate(idleft);
		UsedGate(idright);
	}
}

bool PRFClientSharing::evaluateUNIVGate(GATE* gate) {
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

//#define DEBUGYAOCLIENT
void PRFClientSharing::EvaluateUniversalGate(GATE* gate, uint32_t pos, GATE* gleft, GATE* gright)
{
	BYTE* lkey, * rkey, * okey;
	uint32_t id;
	lkey = gleft->gs.yval + pos * m_nSecParamBytes;
	rkey = gright->gs.yval + pos * m_nSecParamBytes;
	okey = gate->gs.yval + pos * m_nSecParamBytes;

	id = (lkey[m_nSecParamBytes - 1] & 0x01) << 1;
	id += (rkey[m_nSecParamBytes - 1] & 0x01);

	lkey[m_nSecParamBytes - 1] &= 0xFE;
	rkey[m_nSecParamBytes - 1] &= 0xFE;

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
//#undef DEBUGYAOCLIENT

void PRFClientSharing::InitClient()
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
				m_xorAESProcessor = std::make_unique<HybridPRFProcessor<PRFXorLTEvaluatingVaesProcessor, PRFXorLTEvaluatingAesniProcessor>>(getXorQueue(), m_vGates, m_vXorIds);
				m_andAESProcessor = std::make_unique<HybridPRFProcessor<PRFAndLTEvaluatingVaesProcessor, PRFAndLTEvaluatingAesniProcessor>>(getAndQueue(), m_vGates, m_vAndIds);
			}
			else {
				m_xorAESProcessor = std::make_unique<PRFXorLTEvaluatingVaesProcessor>(getXorQueue(), m_vGates, m_vXorIds);
				m_andAESProcessor = std::make_unique<PRFAndLTEvaluatingVaesProcessor>(getAndQueue(), m_vGates, m_vAndIds);
			}
		}
		else if (CPU_FEATURES.aes && CPU_FEATURES.sse4_1) {

			m_xorAESProcessor = std::make_unique<PRFXorLTEvaluatingAesniProcessor>(getXorQueue(), m_vGates, m_vXorIds);
			m_andAESProcessor = std::make_unique<PRFAndLTEvaluatingAesniProcessor>(getAndQueue(), m_vGates, m_vAndIds);
		}
		else
		{
			std::cerr << "unsupported host CPU." << std::endl;
			assert(false);
		}
	}
}
