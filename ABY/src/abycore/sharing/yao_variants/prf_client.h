#ifndef __PRF_CLIENT_H__
#define __PRF_CLIENT_H__

#include "../yaoclientsharing.h"
#include "../aes_processors/aes_processor.h"

class PRFClientSharing : public YaoClientSharing {
public:
	/** Constructor of the class.*/
	PRFClientSharing(e_sharing context, e_role role, uint32_t sharebitlen, ABYCircuit* circuit, crypto* crypt, const std::string& circdir = ABY_CIRCUIT_DIR) :
		YaoClientSharing(context, role, sharebitlen, circuit, crypt, circdir) {
		InitClient();
	}
	/** Destructor of the class.*/
	//~HalfGatesPRPClientSharing();
protected:
	size_t ciphertextPerAND() const override { return 2; }
	size_t ciphertextPerXOR() const override { return 1; }

	void evaluateDeferredXORGates(size_t numWires) override;
	void evaluateDeferredANDGates(size_t numWires) override;
	bool evaluateXORGate(GATE* gate) override { m_vXorIds.push_back(m_nWireCounter); m_nWireCounter += gate->nvals; return false; }
	bool evaluateANDGate(GATE* gate) override { m_vAndIds.push_back(m_nWireCounter); m_nWireCounter += gate->nvals; return false; }
	bool evaluateUNIVGate(GATE* gate) override;

	void resetEvaluationSpecific() override { m_vXorIds.clear(); m_vAndIds.clear(); m_nWireCounter = 0; }
private:
	void InitClient();
	void EvaluateUniversalGate(GATE* gate, uint32_t pos, GATE* gleft, GATE* gright);

	std::unique_ptr<AESProcessor> m_xorAESProcessor;
	std::unique_ptr<AESProcessor> m_andAESProcessor;

	std::vector<uint64_t> m_vXorIds;
	std::vector<uint64_t> m_vAndIds;
	uint64_t m_nWireCounter = 0;
};

#endif