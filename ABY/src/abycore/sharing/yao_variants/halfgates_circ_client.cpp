#include "halfgates_circ_client.h"

#include "../aes_processors/aesni_halfgate_processors.h"
#include "../aes_processors/vaes_halfgate_processors.h"
#include "../cpu_features/include/cpuinfo_x86.h"

static const cpu_features::X86Features CPU_FEATURES = cpu_features::GetX86Info().features;

std::unique_ptr<AESProcessor> HalfGatesCircClientSharing::provideEvaluationProcessor() const
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
				return std::make_unique<HybridHalfgateEvaluatingProcessor<InputKeyLTEvaluatingVaesProcessor, InputKeyLTEvaluatingAesniProcessor>>(getAndQueue(), m_vGates);
			}
			else {
				return std::make_unique<InputKeyLTEvaluatingVaesProcessor>(getAndQueue(), m_vGates);
			}
		}
		else if (CPU_FEATURES.aes && CPU_FEATURES.sse4_1)
			return std::make_unique<InputKeyLTEvaluatingAesniProcessor>(getAndQueue(), m_vGates);
		else
		{
			std::cerr << "unsupported host CPU." << std::endl;
			assert(false);
		}
	}
}
