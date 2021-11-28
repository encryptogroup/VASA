#include "halfgates_mi_server.h"

#include "../aes_processors/aesni_halfgate_processors.h"
#include "../aes_processors/vaes_halfgate_processors.h"
#include "../cpu_features/include/cpuinfo_x86.h"
#include "../../aby/abysetup.h"

#include <iostream>

static const cpu_features::X86Features CPU_FEATURES = cpu_features::GetX86Info().features;

std::unique_ptr<AESProcessorHalfGateGarbling> HalfGatesMIServerSharing::provideGarblingProcessor() const {
	if (m_nSecParamBytes != 16)
	{
		std::cerr << "unsupported security parameter." << std::endl;
		assert(false);
	}
	else
	{
		if (ENABLE_VAES && CPU_FEATURES.vaes && CPU_FEATURES.avx512f && CPU_FEATURES.avx512bw && CPU_FEATURES.avx512vl) {
			return std::make_unique<MILTGarblingVaesProcessor>(getAndQueue(), m_vGates);
		}
		else if (CPU_FEATURES.aes && CPU_FEATURES.sse4_1)
			return std::make_unique<MILTGarblingAesniProcessor>(getAndQueue(), m_vGates);
		else
		{
			std::cerr << "unsupported host CPU." << std::endl;
			assert(false);
		}
	}
}

void HalfGatesMIServerSharing::prepareGarblingSpecificSetup()
{
	HalfGatesPRPServerSharing::prepareGarblingSpecificSetup();
	m_vIdxOffset.Create(m_cCrypto->get_seclvl().symbits, m_cCrypto);

	AESProcessorHalfGateGarbling* processor = m_aesProcessor.get();
	RandomizedProcessor* casted = dynamic_cast<RandomizedProcessor*>(processor);
	casted->setUniqueRandomValue(m_vIdxOffset.GetArr());
}

void HalfGatesMIServerSharing::sendDataGarblingSpecific(ABYSetup* setup)
{
	setup->AddSendTask(m_vIdxOffset.GetArr(),16);
}
