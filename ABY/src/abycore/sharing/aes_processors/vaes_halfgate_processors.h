#ifndef __VAES_HALFGATE_PROCESSORS_H__
#define __VAES_HALFGATE_PROCESSORS_H__

#include <vector>
#include <memory>
#include <cstdint>

using std::uint8_t;

#include "../../circuit/abycircuit.h"
#include "aes_processor.h"
#include "../../ABY_utils/memory.h"
#include "../yaoserversharing.h"
#include "aesni_halfgate_processors.h"

class FixedKeyLTGarblingVaesProcessor : public AESProcessorHalfGateGarbling
{
public:
	FixedKeyLTGarblingVaesProcessor(const std::vector<GATE*>& tableGateQueue, const std::vector<GATE>& vGates) :
		m_tableGateQueue(tableGateQueue),
		m_vGates(vGates)
	{
	}
	virtual void setGlobalKey(const uint8_t* r) override { m_globalRandomOffset = r;}
	virtual void computeAESOutKeys(uint32_t tableCounter, size_t numTablesInBatch, uint8_t* tableBuffer) override;
private:
	template<class M, class R>
	friend class HybridHalfgateGarblingProcessor;
	// only processes multiples of width
	template<size_t width> void computeOutKeysAndTable(uint32_t tableCounter, size_t numTablesInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer);

	FixedKeyProvider m_fixedKeyProvider;
	const std::vector<GATE*>& m_tableGateQueue;
	const std::vector<GATE>& m_vGates;
	const uint8_t* m_globalRandomOffset;

	size_t vectorWidth() const override;
	void BulkProcessor(uint32_t wireCounter, size_t numWiresInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer) override;
	void LeftoversProcessor(uint32_t wireCounter, size_t numWiresInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer) override;
};

class FixedKeyLTEvaluatingVaesProcessor : public AESProcessor
{
public:
	FixedKeyLTEvaluatingVaesProcessor(const std::vector<GATE*>& gateQueue, const std::vector<GATE>& vGates) :
		m_gateQueue(gateQueue),
		m_vGates(vGates)
	{
	}
	virtual void computeAESOutKeys(uint32_t tableCounter, size_t numTablesInBatch, uint8_t* receivedTables) override;
private:
	template<class M, class R>
	friend class HybridHalfgateEvaluatingProcessor;
	template<size_t width>  void computeAESOutKeys(uint32_t tableCounter, size_t queueStartIndex, size_t simdStartOffset, size_t numTablesInBatch, const uint8_t* receivedTables);

	FixedKeyProvider m_fixedKeyProvider;
	const std::vector<GATE*>& m_gateQueue;
	const std::vector<GATE>& m_vGates;

	size_t vectorWidth() const override;
	void BulkProcessor(uint32_t wireCounter, size_t numWiresInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer) override;
	void LeftoversProcessor(uint32_t wireCounter, size_t numWiresInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer) override;
};

class InputKeyLTGarblingVaesProcessor : public AESProcessorHalfGateGarbling
{
public:
	InputKeyLTGarblingVaesProcessor(const std::vector<GATE*>& tableGateQueue, const std::vector<GATE>& vGates) :
		m_tableGateQueue(tableGateQueue),
		m_vGates(vGates)
	{
	}
	virtual void setGlobalKey(const uint8_t* r) override { m_globalRandomOffset = r; }
	virtual void computeAESOutKeys(uint32_t tableCounter, size_t numTablesInBatch, uint8_t* tableBuffer) override;
private:
	template<class M, class R>
	friend class HybridHalfgateGarblingProcessor;
	// only processes multiples of width
	template<size_t width> void computeOutKeysAndTable(uint32_t tableCounter, size_t numTablesInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer);

	const std::vector<GATE*>& m_tableGateQueue;
	const std::vector<GATE>& m_vGates;
	const uint8_t* m_globalRandomOffset;

	size_t vectorWidth() const override;
	void BulkProcessor(uint32_t wireCounter, size_t numWiresInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer) override;
	void LeftoversProcessor(uint32_t wireCounter, size_t numWiresInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer) override;
};

class InputKeyLTEvaluatingVaesProcessor : public AESProcessor
{
public:
	InputKeyLTEvaluatingVaesProcessor(const std::vector<GATE*>& gateQueue, const std::vector<GATE>& vGates) :
		m_gateQueue(gateQueue),
		m_vGates(vGates)
	{
	}
	virtual void computeAESOutKeys(uint32_t tableCounter, size_t numTablesInBatch, uint8_t* receivedTables) override;
private:
	template<class M, class R>
	friend class HybridHalfgateEvaluatingProcessor;
	template<size_t width>  void computeAESOutKeys(uint32_t tableCounter, size_t queueStartIndex, size_t simdStartOffset, size_t numTablesInBatch, const uint8_t* receivedTables);

	const std::vector<GATE*>& m_gateQueue;
	const std::vector<GATE>& m_vGates;

	size_t vectorWidth() const override;
	void BulkProcessor(uint32_t wireCounter, size_t numWiresInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer) override;
	void LeftoversProcessor(uint32_t wireCounter, size_t numWiresInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer) override;
};

class MILTGarblingVaesProcessor : public AESProcessorHalfGateGarbling, public RandomizedProcessor
{
public:
	MILTGarblingVaesProcessor(const std::vector<GATE*>& tableGateQueue, const std::vector<GATE>& vGates) :
		m_tableGateQueue(tableGateQueue),
		m_vGates(vGates)
	{
	}
	virtual void setGlobalKey(const uint8_t* r) override { m_globalRandomOffset = r; }
	virtual void setUniqueRandomValue(const uint8_t* val) override { m_globalIndexOffset = val; }
	virtual void computeAESOutKeys(uint32_t tableCounter, size_t numTablesInBatch, uint8_t* tableBuffer) override;
private:
	template<class M, class R>
	friend class HybridHalfgateGarblingProcessor;
	// only processes multiples of width
	template<size_t width> void computeOutKeysAndTable(uint32_t tableCounter, size_t numTablesInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer);

	const std::vector<GATE*>& m_tableGateQueue;
	const std::vector<GATE>& m_vGates;
	const uint8_t* m_globalRandomOffset;
	const uint8_t* m_globalIndexOffset;

	size_t vectorWidth() const override;
	void BulkProcessor(uint32_t wireCounter, size_t numWiresInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer) override;
	void LeftoversProcessor(uint32_t wireCounter, size_t numWiresInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer) override;
};

class MILTEvaluatingVaesProcessor : public AESProcessor, public RandomizedProcessor
{
public:
	MILTEvaluatingVaesProcessor(const std::vector<GATE*>& gateQueue, const std::vector<GATE>& vGates) :
		m_gateQueue(gateQueue),
		m_vGates(vGates)
	{
	}
	virtual void setUniqueRandomValue(const uint8_t* val) override { m_globalIndexOffset = val; }
	virtual void computeAESOutKeys(uint32_t tableCounter, size_t numTablesInBatch, uint8_t* receivedTables) override;
private:
	template<class M, class R>
	friend class HybridHalfgateEvaluatingProcessor;
	template<size_t width>  void computeAESOutKeys(uint32_t tableCounter, size_t queueStartIndex, size_t simdStartOffset, size_t numTablesInBatch, const uint8_t* receivedTables);

	const std::vector<GATE*>& m_gateQueue;
	const std::vector<GATE>& m_vGates;
	const uint8_t* m_globalIndexOffset;

	size_t vectorWidth() const override;
	void BulkProcessor(uint32_t wireCounter, size_t numWiresInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer) override;
	void LeftoversProcessor(uint32_t wireCounter, size_t numWiresInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer) override;
};



#endif