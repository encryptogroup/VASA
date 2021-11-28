# EMP-Tool

You may read the original readme with build and usage instructions as [`README-ORIGINAL.md`](README-ORIGINAL.md)

## Main Changes

|Keywords|File|Function|Summary|
|-|-|-|-|
|Larger Batch Sizes|[`constants.h`](emp-tool/utils/constants.h)|`AES_BATCH_SIZE`|Raised the AES batch size from 8 to 16 to accomodate VAES' parallel block processing.|
|AVX512 transposition|[`block.h`](emp-tool/utils/block.h)|`sse_trans()`|Expanded the main 16x8 transposition loop to 64x8 using analogous AVX512 instructions as a preprocessing step.|
|Addition-based PRG|[`prg.h`](emp-tool/utils/prg.h)|`PRG::random_block()`|Replaced explicitly setting the CTR PRG AES inputs using a general-purpose register with a vectorized addition.|
|VAES ECB|[`aes.h`](emp-tool/utils/aes.h), [`aes_opt.h`](emp-tool/utils/aes_opt.h)|`AES_opt_key_schedule()`, `ParaEnc()`, `AES_ecb_encrypt_blks()`|Added static / dynamic choice of AES-NI vs VAES depending on the amount of requested operations.|
