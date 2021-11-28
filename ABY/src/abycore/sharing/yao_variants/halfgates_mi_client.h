#ifndef __HALFGATES_MI_CLIENT_H__
#define __HALFGATES_MI_CLIENT_H__

#include "halfgates_prp_client.h"

class HalfGatesMIClientSharing : public HalfGatesPRPClientSharing {
public:
	/** Constructor of the class.*/
	HalfGatesMIClientSharing(e_sharing context, e_role role, uint32_t sharebitlen, ABYCircuit* circuit, crypto* crypt, const std::string& circdir = ABY_CIRCUIT_DIR) :
		HalfGatesPRPClientSharing(context, role, sharebitlen, circuit, crypt, circdir)
	{
		m_aesProcessor = provideEvaluationProcessor();
	}
protected:
	void receiveDataGarblingSpecific(ABYSetup* setup) override;
private:
	std::unique_ptr<AESProcessor> provideEvaluationProcessor() const;
	CBitVector m_vIdxOffset;
};

#endif