#ifndef __AESNI_PRF_PROCESSORS_H__
#define __AESNI_PRF_PROCESSORS_H__

#include "aes_processor.h"

// handles evaluating queued up XOR gates
class PRFXorLTEvaluatingAesniProcessor : public AESProcessor
{
public:
	PRFXorLTEvaluatingAesniProcessor(const std::vector<GATE*>& gateQueue, const std::vector<GATE>& vGates, const std::vector<uint64_t>& wireIds) :
		m_gateQueue(gateQueue),
		m_vGates(vGates),
		m_vWireIds(wireIds)
	{
	}
	virtual void computeAESOutKeys(uint32_t tableCounter, size_t numTablesInBatch, uint8_t* receivedTables) override;
private:
	template<class M, class R>
	friend class HybridPRFProcessor;
	template<size_t width>  
	void computeAESOutKeys(uint32_t tableCounter, size_t queueStartIndex, size_t simdStartOffset, size_t numTablesInBatch, const uint8_t* receivedTables);

	const std::vector<GATE*>& m_gateQueue;
	const std::vector<GATE>& m_vGates;
	const std::vector<uint64_t>& m_vWireIds;

	size_t vectorWidth() const override;
	void BulkProcessor(uint32_t wireCounter, size_t numWiresInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer) override;
	void LeftoversProcessor(uint32_t wireCounter, size_t numWiresInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer) override;
};

// handles garbling queued up XOR gates
class PRFXorLTGarblingAesniProcessor : public AESProcessor
{
public:
	PRFXorLTGarblingAesniProcessor(const std::vector<GATE*>& gateQueue, const std::vector<GATE>& vGates, const std::vector<uint64_t>& wireIds) :
		m_gateQueue(gateQueue),
		m_vGates(vGates),
		m_vWireIds(wireIds)
	{
	}
	virtual void computeAESOutKeys(uint32_t tableCounter, size_t numTablesInBatch, uint8_t* receivedTables) override;
private:
	template<class M, class R>
	friend class HybridPRFProcessor;
	template<size_t width>  
	void computeAESOutKeys(uint32_t tableCounter, size_t queueStartIndex, size_t simdStartOffset, size_t numTablesInBatch, uint8_t* receivedTables);

	const std::vector<GATE*>& m_gateQueue;
	const std::vector<GATE>& m_vGates;
	const std::vector<uint64_t>& m_vWireIds;

	size_t vectorWidth() const override;
	void BulkProcessor(uint32_t wireCounter, size_t numWiresInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer) override;
	void LeftoversProcessor(uint32_t wireCounter, size_t numWiresInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer) override;
};

// handles evaluating queued up AND gates
class PRFAndLTEvaluatingAesniProcessor : public AESProcessor
{
public:
	PRFAndLTEvaluatingAesniProcessor(const std::vector<GATE*>& gateQueue, const std::vector<GATE>& vGates, const std::vector<uint64_t>& wireIds) :
		m_gateQueue(gateQueue),
		m_vGates(vGates),
		m_vWireIds(wireIds)
	{
	}
	virtual void computeAESOutKeys(uint32_t tableCounter, size_t numTablesInBatch, uint8_t* receivedTables) override;
private:
	template<class M, class R>
	friend class HybridPRFProcessor;
	template<size_t width>  
	void computeAESOutKeys(uint32_t tableCounter, size_t queueStartIndex, size_t simdStartOffset, size_t numTablesInBatch, const uint8_t* receivedTables);

	const std::vector<GATE*>& m_gateQueue;
	const std::vector<GATE>& m_vGates;
	const std::vector<uint64_t>& m_vWireIds;

	size_t vectorWidth() const override;
	void BulkProcessor(uint32_t wireCounter, size_t numWiresInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer) override;
	void LeftoversProcessor(uint32_t wireCounter, size_t numWiresInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer) override;
};

// handles garbling queued up AND gates
class PRFAndLTGarblingAesniProcessor : public AESProcessor
{
public:
	PRFAndLTGarblingAesniProcessor(const std::vector<GATE*>& gateQueue, const std::vector<GATE>& vGates, const std::vector<uint64_t>& wireIds) :
		m_gateQueue(gateQueue),
		m_vGates(vGates),
		m_vWireIds(wireIds)
	{
	}
	virtual void computeAESOutKeys(uint32_t tableCounter, size_t numTablesInBatch, uint8_t* receivedTables) override;
private:
	template<class M, class R>
	friend class HybridPRFProcessor;
	template<size_t width>
	void computeAESOutKeys(uint32_t tableCounter, size_t queueStartIndex, size_t simdStartOffset, size_t numTablesInBatch, uint8_t* receivedTables);

	const std::vector<GATE*>& m_gateQueue;
	const std::vector<GATE>& m_vGates;
	const std::vector<uint64_t>& m_vWireIds;

	size_t vectorWidth() const override;
	void BulkProcessor(uint32_t wireCounter, size_t numWiresInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer) override;
	void LeftoversProcessor(uint32_t wireCounter, size_t numWiresInBatch, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer) override;
};

#endif
