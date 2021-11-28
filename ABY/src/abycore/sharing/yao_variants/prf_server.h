#ifndef __PRF_SERVER_H__
#define __PRF_SERVER_H__

#include "../yaoserversharing.h"
#include "../aes_processors/aes_processor.h"
#include "../aes_processors/aesni_halfgate_processors.h"

class PRFServerSharing : public YaoServerSharing {
public:
	/** Constructor of the class.*/
	PRFServerSharing(e_sharing context, e_role role, uint32_t sharebitlen, ABYCircuit* circuit, crypto* crypt, const std::string& circdir = ABY_CIRCUIT_DIR) :
		YaoServerSharing(context, role, sharebitlen, circuit, crypt, circdir) {
		InitServer();
	}
	/** Destructor of the class.*/
	//~HalfGatesPRPClientSharing();
protected:
	size_t ciphertextPerAND() const override { return 2; }
	size_t ciphertextPerXOR() const override { return 1; }

	// we need these functions
	// because for regular freeXOR the invariant is 
	// that lowBit(key[0]) XOR pi == representedValue
	// whereas this PRF garbling uses the invariant that key[0] is always the 0 wire key
	bool evaluateConstantGate(GATE* gate) override;
	void createOppositeInputKeys(CBitVector& oppositeInputKeys, CBitVector& reglarInputKeys, size_t numKeys) override;
	void copyServerInputKey(uint8_t inputBit, uint8_t permutationBit, size_t targetByteOffset, size_t sourceByteOffset) override;
	uint8_t computePermutationValueFromBoolConv(uint8_t inputBit, uint8_t permutationBit) override { return inputBit; }
	void prepareInputkeysConversion(CBitVector& keys, size_t offset, uint8_t permutationBit) override { keys.XORByte(offset + m_nSecParamBytes - 1, permutationBit); }
	void prepareGarblingSpecificSetup() override {}

	void evaluateDeferredXORGates(size_t numWires) override;
	void evaluateDeferredANDGates(ABYSetup* setup, size_t numWires) override;
	bool evaluateXORGate(GATE* gate) override { m_vXorIds.push_back(m_nWireCounter); m_nWireCounter += gate->nvals; return false; }
	bool evaluateANDGate(ABYSetup*, GATE* gate) override { m_vAndIds.push_back(m_nWireCounter); m_nWireCounter += gate->nvals; return false; }
	bool evaluateUNIVGate(GATE* gate) override;
	bool evaluateInversionGate(GATE* gate) override;

	void resetGarblingSpecific() override { m_vXorIds.clear(); m_vAndIds.clear(); m_nWireCounter = 0; }
private:
	void InitServer();
	void choosePi(GATE* gate);
	void GarbleUniversalGate(GATE* ggate, uint32_t pos, GATE* gleft, GATE* gright, uint32_t ttable);

	std::unique_ptr<AESProcessor> m_xorAESProcessor;
	std::unique_ptr<AESProcessor> m_andAESProcessor;

	std::unique_ptr<FixedKeyProvider> m_andPiBitProvider;
	uint64_t piCounter=0;

	std::vector<uint64_t> m_vXorIds;
	std::vector<uint64_t> m_vAndIds;
	uint64_t m_nWireCounter =0;
};

#endif