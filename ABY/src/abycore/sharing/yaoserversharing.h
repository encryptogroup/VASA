/**
 \file 		yaoserversharing.h
 \author	michael.zohner@ec-spride.de
 \copyright	ABY - A Framework for Efficient Mixed-protocol Secure Two-party Computation
			Copyright (C) 2019 Engineering Cryptographic Protocols Group, TU Darmstadt
			This program is free software: you can redistribute it and/or modify
            it under the terms of the GNU Lesser General Public License as published
            by the Free Software Foundation, either version 3 of the License, or
            (at your option) any later version.
            ABY is distributed in the hope that it will be useful,
            but WITHOUT ANY WARRANTY; without even the implied warranty of
            MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
            GNU Lesser General Public License for more details.
            You should have received a copy of the GNU Lesser General Public License
            along with this program. If not, see <http://www.gnu.org/licenses/>.
 \brief		Yao Server Sharing class.
 */

#ifndef __YAOSERVERSHARING_H__
#define __YAOSERVERSHARING_H__

#include "sharing.h"
#include <algorithm>
#include <deque>
#include <vector>
#include <memory>
#include "yaosharing.h"
#include "../ABY_utils/memory.h"
#include "aes_processors/aes_processor.h"


//#define DEBUGYAOSERVER
/**
 Yao Server Sharing class.
 */
class YaoServerSharing: public YaoSharing {

public:
	/**
	 Constructor of the class.
	 */
	YaoServerSharing(e_sharing context, e_role role, uint32_t sharebitlen, ABYCircuit* circuit, crypto* crypt, const std::string& circdir = ABY_CIRCUIT_DIR) :
			YaoSharing(context, role, sharebitlen, circuit, crypt, circdir) {
		InitServer();
	}
	;
	/**
	 Destructor of the class.
	 */
	virtual ~YaoServerSharing();

	//MEMBER FUNCTIONS FROM SUPER CLASS YAO SHARING
	void Reset();
	void PrepareSetupPhase(ABYSetup* setup);
	void PerformSetupPhase(ABYSetup* setup);
	void FinishSetupPhase(ABYSetup* setup);
	void EvaluateLocalOperations(uint32_t level);
	void EvaluateInteractiveOperations(uint32_t level);
	void SendConversionValues(uint32_t gateid);

	void FinishCircuitLayer();

	void PrepareOnlinePhase();

	void InstantiateGate(GATE* gate);

	void GetDataToSend(std::vector<BYTE*>& sendbuf, std::vector<uint64_t>& bytesize);
	void GetBuffersToReceive(std::vector<BYTE*>& rcvbuf, std::vector<uint64_t>& rcvbytes);

	uint32_t AssignInput(CBitVector& input);
	uint32_t GetOutput(CBitVector& out);

	const char* sharing_type() {
		return "Yao server";
	}
	;
	//ENDS HERE..
protected:
	virtual void evaluateDeferredANDGates(ABYSetup* setup, size_t numWires) = 0;
	virtual void evaluateDeferredXORGates(size_t numWires) = 0;

	// these return true if they already fully processed the gate
	// at hand which then does not need to be queued
	virtual bool evaluateXORGate(GATE* gate) = 0;
	virtual bool evaluateANDGate(ABYSetup* setup, GATE* gate) = 0;
	virtual bool evaluateUNIVGate(GATE* gate) = 0;
	virtual bool evaluateConstantGate(GATE* gate) = 0;
	virtual bool evaluateInversionGate(GATE* gate) = 0;

	virtual void createOppositeInputKeys(CBitVector& oppositeInputKeys, CBitVector& reglarInputKeys, size_t numKeys) =0;
	virtual void copyServerInputKey(uint8_t inputBit, uint8_t permutationBit, size_t targetByteOffset, size_t sourceByteOffset) = 0;
	virtual uint8_t computePermutationValueFromBoolConv(uint8_t inputBit, uint8_t permutationBit) = 0;
	virtual void prepareInputkeysConversion(CBitVector& keys, size_t offset, uint8_t permutationBit) {}


	virtual void prepareGarblingSpecificSetup() = 0;
	virtual void resetGarblingSpecific() = 0;
	virtual void sendDataGarblingSpecific(ABYSetup* setup) {}

	const std::vector<GATE*>& getAndQueue() const { return m_andQueue; }
	const std::vector<GATE*>& getXorQueue() const { return m_xorQueue; }
	const CBitVector& getOppositeServerInputKeys() const { return m_vOppositeServerInputKeys; }

	// this is only for AND gates
	uint64_t m_nGarbledTableSndCtr;
	
	CBitVector m_vServerKeySndBuf; /**< Server Key Sender Buffer*/
private:
	std::vector<GATE*> m_andQueue; /**< Pointer to the current layer's AND gates, one entry per garbled table (i.e. one per bit which means >1 per SIMD AND)*/
	std::vector<GATE*> m_xorQueue;

	CBitVector m_vOppositeServerInputKeys;
	CBitVector m_vOppositeClientInputKeys;
	bool m_inDestructor = false;
	
	//Permutation bits for the servers input keys
	CBitVector m_vPermBits; /**< _____________*/
	//Random values from output of ot extension
	std::vector<CBitVector> m_vROTMasks; /**< Masks_______________*/
	uint32_t m_nClientInputKexIdx; /**< Client __________*/
	uint32_t m_nClientInputKeyCtr; /**< Client __________*/

	std::vector<CBitVector> m_vClientKeySndBuf; /**< Client Key Sender Buffer*/
	CBitVector m_vClientROTRcvBuf; /**< Client ______________*/

	//std::vector<CBitVector> m_vClientConversionKeySndBuf;
	//CBitVector			m_vClientCOnversionROTRcvBuf;

	CBitVector m_vOutputShareSndBuf; /**< Output Share Sender Buffer.*/
	CBitVector m_vOutputShareRcvBuf; /**< Output Share Receiver Buffer.*/

	std::vector<GATE*> m_vServerOutputGates; /**< Server Output Gates*/

	uint32_t m_nOutputShareRcvCtr; /**< Output Share Receiver Counter*/

	uint64_t m_nPermBitCtr; /**< _____________*/
	uint64_t m_nServerInBitCtr; /**< _____________*/

	uint32_t m_nServerKeyCtr; /**< _____________*/
	uint32_t m_nClientInBitCtr; /**< _____________*/

	//CBitVector

	std::vector<uint32_t> m_vClientInputGate; /**< _____________*/
	std::deque<input_gate_val_t> m_vPreSetInputGates;/**< _____________*/
	std::deque<a2y_gate_pos_t> m_vPreSetA2YPositions;/**< _____________*/
	e_role* m_vOutputDestionations; /** <  _____________*/
	uint32_t m_nOutputDestionationsCtr;


	//std::deque<uint32_t> 			m_vClientInputGate;

	/**Initialising the server. */
	void InitServer();
	/**Initialising a new layer.*/
	void InitNewLayer();

	/**
	 Creating Random wire keys.
	 \param	vec 		Bit vector.
	 \param 	numkeys		number of keys.
	 */
	void CreateRandomWireKeys(CBitVector& vec, uint32_t numkeys);
	/**
	 Creating and sending Garbled Circuit.
	 \param 	setup 	ABYSetup Object.
	 */
	void CreateAndSendGarbledCircuit(ABYSetup* setup);
	/**
	 Receiving the garbled circuit object.
	 */
	void ReceiveGarbledCircuit();
	/**
	 Method for evaluating a Input gate for the inputted
	 gate id.
	 \param gateid	Gate Identifier
	 */
	void EvaluateInputGate(uint32_t gateid);
	/**
	 Method for evaluating XOR gate for the inputted
	 gate object.
	 \param gate		Gate Object
	 */
	void EvaluateXORGate(GATE* gate);
	/**
	 Method for evaluating AND gate for the inputted
	 gate object.
	 \param gate		Gate Object
	 */
	void EvaluateANDGate(GATE* gate, ABYSetup* setup);
	/**
	 Method for evaluating a Universal gate for the inputted
	 gate object.
	 \param gate		Gate Object
	 */
	void EvaluateUniversalGate(GATE* gate);
	/**
	 Method for evaluating SIMD gate for the inputted
	 gateid.
	 \param gateid		Gate identifier
	 */
	void EvaluateSIMDGate(uint32_t gateid);
	/**
	 Method for evaluating Inversion gate for the inputted
	 gate object.
	 \param gate		Gate Object
	 */
	void EvaluateInversionGate(GATE* gate);
	/**
	 Method for evaluating conversion gate for the inputted
	 gateid.
	 \param gateid		Gate Identifier
	 */
	void EvaluateConversionGate(uint32_t gateid);
	/**
	 Method for garbling a universal gate.
	 \param ggate	gate Object.
	 \param pos 		Position of the object in the queue.
	 \param gleft	left gate in the queue.
	 \param gright	right gate in the queue.
	 \param ttable	the 4-bit truth table of the form x_0x_1x_2x_3
	 */
	void GarbleUniversalGate(GATE* ggate, uint32_t pos, GATE* gleft, GATE* gright, uint32_t ttable);
	/**
	 Method for creating garbled table.
	 \param ggate	gate Object.
	 \param pos 		Position of the object in the queue.
	 \param gleft	left gate in the queue.
	 \param gright	right gate in the queue.
	 */
	void CreateGarbledTablePrepared(GATE* ggate, uint32_t pos, GATE* gleft, GATE* gright);
	/**
	 PrecomputeGC______________
	 \param queue 	Dequeue Object.
	 \param setup	Is needed to perform pipelined sending of the circuit
	 */
	void PrecomputeGC(std::deque<uint32_t>& queue, ABYSetup* setup);

	//void EvaluateClientOutputGate(GATE* gate);
	void CollectClientOutputShares();
	/**
	 Method for evaluating a constant gate for the inputted
	 gate id.
	 \param gateid	Gate Identifier
	 */
	void EvaluateConstantGate(GATE* gate);
	/**
	 Method for evaluating Output gate for the inputted
	 gate object.
	 \param gate		Gate Object
	 */
	void EvaluateOutputGate(GATE* gate);

	/**
	 Send Server Keys from the given gateid.
	 \param	gateid 	Gate Identifier
	 */
	void SendServerInputKey(uint32_t gateid);
	/**
	 Send Client Keys from the given gateid.
	 \param	gateid 	Gate Identifier
	 */
	void SendClientInputKey(uint32_t gateid);
	/**
	 Method for assigning Output shares.
	 */
	void AssignOutputShares();

	/**
	* Evaluates the AND gate by computing the wire keys / mask buffers just in time before the table evaluation
	* \param ggate the gate for which the wire keys are to be computed
	* \param pos the SIMD position
	* \param gleft left parent gate
	* \param gright right parent gate
	*/
	void CreateGarbledTableJIT(GATE* ggate, uint32_t pos,GATE* gleft,GATE* gright);


	/**
	* Evaluates the AND garbled tables queued up in m_andQueue and sends them off
	* \param setup used to send the garbled tables off whenever a batch is full
	*/
	void EvaluateDeferredANDGates(ABYSetup* setup);

	/**
	* Checks whether the gate with the given id is an AND gate that has not yet been processed
	* and thus would cause an error if referenced. Returns true if such an error would be produced.
	* \param gateid the gate to be checked
	*/
	bool CheckIfGateTrapsANDGates(uint32_t gateid) const;
};

#endif /* __YAOSERVERSHARING_H__ */
