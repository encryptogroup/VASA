#ifndef __HALFGATES_CIRC_CLIENT_H__
#define __HALFGATES_CIRC_CLIENT_H__

#include "halfgates_prp_client.h"

class HalfGatesCircClientSharing : public HalfGatesPRPClientSharing {
public:
	/** Constructor of the class.*/
	HalfGatesCircClientSharing(e_sharing context, e_role role, uint32_t sharebitlen, ABYCircuit* circuit, crypto* crypt, const std::string& circdir = ABY_CIRCUIT_DIR) :
		HalfGatesPRPClientSharing(context, role, sharebitlen, circuit, crypt, circdir)
	{
		m_aesProcessor = provideEvaluationProcessor();
	}
private:
	std::unique_ptr<AESProcessor> provideEvaluationProcessor() const;
};

#endif