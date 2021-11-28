#include "halfgates_mi_client.h"

#include "../aes_processors/aesni_halfgate_processors.h"
#include "../aes_processors/vaes_halfgate_processors.h"
#include "../cpu_features/include/cpuinfo_x86.h"
#include "../../aby/abysetup.h"

static const cpu_features::X86Features CPU_FEATURES = cpu_features::GetX86Info().features;

void HalfGatesMIClientSharing::receiveDataGarblingSpecific(ABYSetup* setup)
{
	m_vIdxOffset.Create(m_cCrypto->get_seclvl().symbits);
	setup->AddReceiveTask(m_vIdxOffset.GetArr(), 16);

	// the data is not immediately available
	// however we're only copying a pointer here
	// so by the time its needed the buffer will have been filled
	// similar to how the garbled tables will be there

	AESProcessor* processor = m_aesProcessor.get();
	RandomizedProcessor* casted = dynamic_cast<RandomizedProcessor*>(processor);
	casted->setUniqueRandomValue(m_vIdxOffset.GetArr());
}

std::unique_ptr<AESProcessor> HalfGatesMIClientSharing::provideEvaluationProcessor() const
{
	if (m_nSecParamBytes != 16)
	{
		std::cerr << "unsupported security parameter." << std::endl;
		assert(false);
	}
	else
	{
		if (ENABLE_VAES && CPU_FEATURES.vaes && CPU_FEATURES.avx512f && CPU_FEATURES.avx512bw && CPU_FEATURES.avx512vl) {
			return std::make_unique<MILTEvaluatingVaesProcessor>(getAndQueue(), m_vGates);
		}
		else if (CPU_FEATURES.aes && CPU_FEATURES.sse4_1)
			return std::make_unique<MILTEvaluatingAesniProcessor>(getAndQueue(), m_vGates);
		else
		{
			std::cerr << "unsupported host CPU." << std::endl;
			assert(false);
		}
	}
}
