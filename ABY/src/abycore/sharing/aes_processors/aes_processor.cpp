#include "aes_processor.h"

#include <cassert>

void VectorizedQueueProcessor::ProcessQueue(const std::vector<GATE*>& queue, const size_t numWires, uint32_t wireCounter, uint8_t* tableBuffer)
{
	assert(numWires >= queue.size());
	if (queue.size() == 0)
		return;

	const size_t leftovers = numWires % vectorWidth();
	const size_t mainBulkSize = numWires - leftovers;

	BulkProcessor(wireCounter,mainBulkSize,0,0,tableBuffer);

	size_t numWiresLeft = 0;
	int64_t ridx;

	for (ridx = queue.size() - 1; ridx >= 0; --ridx)
	{
		numWiresLeft += queue[ridx]->nvals;
		if (numWiresLeft >= leftovers)
			break;
	}

	if (leftovers > 0)
	{
		LeftoversProcessor(wireCounter + mainBulkSize, leftovers, ridx, numWiresLeft - leftovers, tableBuffer);
	}
}

void VectorizedQueueProcessor::ProcessLeftovers(const std::vector<GATE*>& queue, uint32_t wireCounter, size_t queueStartIndex, size_t simdStartOffset, uint8_t* tableBuffer)
{
	if (queue.size() == 0)
		return;

	int64_t numWiresLeft = -simdStartOffset;
	for (size_t i = queueStartIndex; i < queue.size(); ++i)
		numWiresLeft += queue[i]->nvals;

	const size_t actualLeftovers = numWiresLeft % vectorWidth();
	const size_t mainBulkSize = numWiresLeft - actualLeftovers;

	BulkProcessor(wireCounter, mainBulkSize, queueStartIndex, simdStartOffset, tableBuffer);

	size_t numWiresLeftCleanup = 0;
	int64_t ridx;

	for (ridx = queue.size() - 1; ridx >= 0; --ridx)
	{
		numWiresLeftCleanup += queue[ridx]->nvals;
		if (numWiresLeftCleanup >= actualLeftovers)
			break;
	}

	if (actualLeftovers > 0)
	{
		LeftoversProcessor(wireCounter + mainBulkSize, actualLeftovers, ridx, numWiresLeftCleanup - actualLeftovers, tableBuffer);
	}
}
