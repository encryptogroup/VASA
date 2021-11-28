#ifndef EMP_CONFIG_H__
#define EMP_CONFIG_H__
namespace emp {
// raised to 16
// to accomodate VAES
// which runs with latency of 3-4 cycles and 4 blocks queued per cycle
// resulting in 12-16 blocks in processing being optimal
const static int AES_BATCH_SIZE = 16;
const static int HASH_BUFFER_SIZE = 1024*8;
const static int NETWORK_BUFFER_SIZE2 = 1024*32;
const static int NETWORK_BUFFER_SIZE = 1024*1024;
const static int FILE_BUFFER_SIZE = 1024*16;
const static int CHECK_BUFFER_SIZE = 1024*8;

const static int XOR = -1;
const static int PUBLIC = 0;
const static int ALICE = 1;
const static int BOB = 2;

const char fix_key[] = "\x61\x7e\x8d\xa2\xa0\x51\x1e\x96\x5e\x41\xc2\x9b\x15\x3f\xc7\x7a";
}
#endif// CONFIG_H__
