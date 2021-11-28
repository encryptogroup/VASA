#ifndef __AES_PROCESSOR_H__
#define __AES_PROCESSOR_H__

#include <cstdint>
#include <cstddef>
#include <vector>
#include <functional>

#include "../../circuit/abycircuit.h"

using std::size_t;


// this design allows specialized implementations for the bulk and leftover cases
// and lifts some of the more mundane tasks like going through the SIMD widths of gates off the implementor
class VectorizedQueueProcessor
{
protected:
	template<class M, class R>
	friend class HybridHalfgateGarblingProcessor;
	template<class M, class R>
	friend class HybridHalfgateEvaluatingProcessor;
	virtual size_t vectorWidth() const = 0;
	void ProcessQueue(const std::vector<GATE*>& queue, const size_t numWires, uint32_t wireCounter, uint8_t* tableBuffer);
	// this is intended to be called by VAES processors to off-load the "small" chunks to more efficient AES-NI engines
	void ProcessLeftovers(const std::vector<GATE*>& queue, uint32_t wireCounter, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer);
	virtual void BulkProcessor(uint32_t wireCounter, size_t numWiresInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer) = 0;
	virtual void LeftoversProcessor(uint32_t wireCounter, size_t numWiresInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer) = 0;
};

class AESProcessor : protected VectorizedQueueProcessor
{
public:
	/**
	* Performs the two AES evaluations per AND gate necessary and writes the result into the right field of the gate
	* \param tableCounter the number of garbled tables inspected at this point
	*/
	virtual void computeAESOutKeys(uint32_t tableCounter, size_t numTablesInBatch, uint8_t* receivedTables) = 0;

	virtual ~AESProcessor() {};
};

class AESProcessorHalfGateGarbling : public AESProcessor
{
public:
	/**
	* Sets the global key difference R for free-XOR garbling
	* \param r the global key difference
	*/
	virtual void setGlobalKey(const uint8_t* r) = 0;

	virtual ~AESProcessorHalfGateGarbling() {};
};

class RandomizedProcessor {
public:
	/**
	* Sets the initial random value, e.g. the initial random CTR offset
	* \param val the initial random value
	*/
	virtual void setUniqueRandomValue(const uint8_t* val) = 0;

	virtual ~RandomizedProcessor() {};
};

// a hybrid implementation that uses the main processor whenever possible and then falls back to the leftover one
// the idea was to evaluate how a hybrid VAES + AES-NI solution would perform
// (Garbling side)
template<class CMainProcessor, class CLeftoversProcessor>
class HybridHalfgateGarblingProcessor : public AESProcessorHalfGateGarbling
{
public:
	HybridHalfgateGarblingProcessor(const std::vector<GATE*>& tableGateQueue, const std::vector<GATE>& vGates)
		:
		m_mainProcessor(tableGateQueue, vGates),
		m_leftoversProcessor(tableGateQueue, vGates),
		m_gateQueue(tableGateQueue)
	{}
	virtual void setGlobalKey(const uint8_t* r) override { m_mainProcessor.setGlobalKey(r); m_leftoversProcessor.setGlobalKey(r); }
	virtual void computeAESOutKeys(uint32_t tableCounter, size_t numTablesInBatch, uint8_t* tableBuffer) override {
		ProcessQueue(m_gateQueue, numTablesInBatch, tableCounter, tableBuffer);
	}

private:
	CMainProcessor m_mainProcessor;
	CLeftoversProcessor m_leftoversProcessor;
	const std::vector<GATE*>& m_gateQueue;

	size_t vectorWidth() const override { return m_mainProcessor.vectorWidth(); }
	void BulkProcessor(uint32_t wireCounter, size_t numWiresInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer) override {
		m_mainProcessor.BulkProcessor(wireCounter, numWiresInBatch, queueStartIndex, simdStartOffset, tableBuffer);
	}
	void LeftoversProcessor(uint32_t wireCounter, size_t numWiresInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer) override {
		m_leftoversProcessor.ProcessLeftovers(m_gateQueue, wireCounter, queueStartIndex, simdStartOffset, tableBuffer);
	}
};

// a hybrid implementation that uses the main processor whenever possible and then falls back to the leftover one
// the idea was to evaluate how a hybrid VAES + AES-NI solution would perform
// (Evaluating side)
template<class CMainProcessor, class CLeftoversProcessor>
class HybridHalfgateEvaluatingProcessor : public AESProcessor
{
public:
	HybridHalfgateEvaluatingProcessor(const std::vector<GATE*>& tableGateQueue, const std::vector<GATE>& vGates)
		:
		m_mainProcessor(tableGateQueue, vGates),
		m_leftoversProcessor(tableGateQueue, vGates),
		m_gateQueue(tableGateQueue)
	{}
	virtual void computeAESOutKeys(uint32_t tableCounter, size_t numTablesInBatch, uint8_t* tableBuffer) override {
		ProcessQueue(m_gateQueue, numTablesInBatch, tableCounter, tableBuffer);
	}

private:
	CMainProcessor m_mainProcessor;
	CLeftoversProcessor m_leftoversProcessor;
	const std::vector<GATE*>& m_gateQueue;

	size_t vectorWidth() const override { return m_mainProcessor.vectorWidth(); }
	void BulkProcessor(uint32_t wireCounter, size_t numWiresInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer) override {
		m_mainProcessor.BulkProcessor(wireCounter, numWiresInBatch, queueStartIndex, simdStartOffset, tableBuffer);
	}
	void LeftoversProcessor(uint32_t wireCounter, size_t numWiresInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer) override {
		m_leftoversProcessor.ProcessLeftovers(m_gateQueue, wireCounter, queueStartIndex, simdStartOffset, tableBuffer);
	}
};

// the same as the above ones just for the PRF-based schemes as they don't have the global key difference
template<class CMainProcessor, class CLeftoversProcessor>
class HybridPRFProcessor : public AESProcessor
{
public:
	HybridPRFProcessor(const std::vector<GATE*>& gateQueue, const std::vector<GATE>& vGates, const std::vector<uint64_t>& wireIds)
		:
		m_mainProcessor(gateQueue, vGates, wireIds),
		m_leftoversProcessor(gateQueue, vGates, wireIds),
		m_gateQueue(gateQueue)
	{}
	virtual void computeAESOutKeys(uint32_t tableCounter, size_t numTablesInBatch, uint8_t* tableBuffer) override {
		ProcessQueue(m_gateQueue, numTablesInBatch, tableCounter, tableBuffer);
	}

private:
	CMainProcessor m_mainProcessor;
	CLeftoversProcessor m_leftoversProcessor;
	const std::vector<GATE*>& m_gateQueue;

	size_t vectorWidth() const override { return m_mainProcessor.vectorWidth(); }
	void BulkProcessor(uint32_t wireCounter, size_t numWiresInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer) override {
		m_mainProcessor.BulkProcessor(wireCounter, numWiresInBatch, queueStartIndex, simdStartOffset, tableBuffer);
	}
	void LeftoversProcessor(uint32_t wireCounter, size_t numWiresInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer) override {
		m_leftoversProcessor.ProcessLeftovers(m_gateQueue, wireCounter, queueStartIndex, simdStartOffset, tableBuffer);
	}
};

#endif