# EMP-AG2PC

You may read the original readme with build and usage instructions as [`README-ORIGINAL.md`](https://github.com/encryptogroup/VASA/blob/master/emp/emp-ag2pc/README-Original.md). Also note that as this library depends on `emp-tool` and `emp-ot` for large parts of the cryptographic processing, optimizations are also relevant here.

## Main Changes

|Keywords|File|Function|Summary|
|-|-|-|-|
|Function-dependent Batching|[`amortized_2pc.h`](emp-ag2pc/amortized_2pc.h)|`AmortizedC2PC::function_dependent_st()`, `AmortizedC2PC::function_dependent_thread()`, `AmortizedC2PC::GarblingHashing()`|Changed to use batched processing using `GarblingHashing` around batched `Hash` (no dependencies for dynamic batching)|
|Online Batching|[`amortized_2pc.h`](emp-ag2pc/amortized_2pc.h)|`AmortizedC2PC::online()`, `AmortizedC2PC::EvaluateANDGates()`| Changed to use dynamic batching with early evaluation around `Hash`.|
