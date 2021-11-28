#ifndef __ABY_UTILS_MEMORY_H__
#define __ABY_UTILS_MEMORY_H__

#include <cstdint>
#include <cstdlib>

struct free_byte_deleter
{
	void operator()(std::uint8_t* p) const {
		std::free(p);
	}
};

#endif