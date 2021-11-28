# CrypTFlow2's "Secure and Correct Inference" (SCI)

You may read the original readme with build and usage instructions as [`README-ORIGINAL.md`](README-ORIGINAL.md)

## Main Changes

|Keywords|File|Function|Summary|
|-|-|-|-|
|Re-Keying AES|[`ccrf.h`](src/utils/ccrf.h)|`CCRF()`|Invocation of the below VAES implementation depending on the amount of blocks for the re-keying AES.|
|AES-CTR|[`prg.h`](src/utils/prg.h)|`PRG128::random_block()`, `PRG256::random_block()`|Invocation of the below VAES AES-CTR implementation depending on the amount of blocks.|
|AES-ECB|[`prp.h`](src/utils/prp.h)|`PRP::permute_block()`|Invocation of the below VAES implementation depending on the amount of blocks.|
|VAES Implementations|[`utils/vaes.h`](src/utils/vaes.h)|All functions in the file| The actual VAES-based implementations of the CTR, ECB, and on-the-fly expanding encryption|
