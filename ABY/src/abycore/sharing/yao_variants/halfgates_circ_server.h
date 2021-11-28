#ifndef __HALFGATES_CIRC_SERVER_H__
#define __HALFGATES_CIRC_SERVER_H__

#include "halfgates_prp_server.h"

class HalfGatesCircServerSharing : public HalfGatesPRPServerSharing {
public:
	/** Constructor of the class.*/
	HalfGatesCircServerSharing(e_sharing context, e_role role, uint32_t sharebitlen, ABYCircuit* circuit, crypto* crypt, const std::string& circdir = ABY_CIRCUIT_DIR) :
		HalfGatesPRPServerSharing(context, role, sharebitlen, circuit, crypt, circdir)
	{
		m_aesProcessor = provideGarblingProcessor();
	}
private:
	std::unique_ptr<AESProcessorHalfGateGarbling> provideGarblingProcessor() const;
};

#endif