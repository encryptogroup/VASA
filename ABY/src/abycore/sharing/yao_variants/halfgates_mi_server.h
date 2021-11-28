#ifndef __HALFGATES_MI_SERVER_H__
#define __HALFGATES_MI_SERVER_H__

#include "halfgates_prp_server.h"

class HalfGatesMIServerSharing : public HalfGatesPRPServerSharing {
public:
	/** Constructor of the class.*/
	HalfGatesMIServerSharing(e_sharing context, e_role role, uint32_t sharebitlen, ABYCircuit* circuit, crypto* crypt, const std::string& circdir = ABY_CIRCUIT_DIR) :
		HalfGatesPRPServerSharing(context, role, sharebitlen, circuit, crypt, circdir)
	{
		m_aesProcessor = provideGarblingProcessor();
	}
protected:
	void prepareGarblingSpecificSetup() override;
	void sendDataGarblingSpecific(ABYSetup* setup) override;
private:
	std::unique_ptr<AESProcessorHalfGateGarbling> provideGarblingProcessor() const;
	CBitVector m_vIdxOffset;
};

#endif