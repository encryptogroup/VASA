#ifndef __HALFGATES_PRP_CLIENT_H__
#define __HALFGATES_PRP_CLIENT_H__

#include "../yaoclientsharing.h"

class HalfGatesPRPClientSharing : public YaoClientSharing {
public:
	/** Constructor of the class.*/
	HalfGatesPRPClientSharing(e_sharing context, e_role role, uint32_t sharebitlen, ABYCircuit* circuit, crypto* crypt, const std::string& circdir = ABY_CIRCUIT_DIR) :
		YaoClientSharing(context, role, sharebitlen, circuit, crypt, circdir),
		m_aesProcessor(provideEvaluationProcessor())
	{
	}
	/** Destructor of the class.*/
	virtual ~HalfGatesPRPClientSharing() {}
protected:
	size_t ciphertextPerAND() const override { return 2; }
	size_t ciphertextPerXOR() const override { return 0; }

	inline void evaluateDeferredXORGates(size_t numWires) override {}
	void evaluateDeferredANDGates(size_t numWires) override;
	bool evaluateXORGate(GATE* gate) override;
	inline bool evaluateANDGate(GATE* gate) override { return false; }
	bool evaluateUNIVGate(GATE* gate) override;

	inline void resetEvaluationSpecific() override {}

	std::unique_ptr<AESProcessor> m_aesProcessor; /**< Processor for the generation of the garbled table PRF calls in a more optimized way*/
private:
	std::unique_ptr<AESProcessor> provideEvaluationProcessor() const;
	void EvaluateGarbledTablePrepared(GATE* gate, uint32_t pos, GATE* gleft, GATE* gright);
	void EvaluateUniversalGate(GATE* gate, uint32_t pos, GATE* gleft, GATE* gright);

	
};

#endif