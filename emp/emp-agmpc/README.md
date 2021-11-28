# EMP-AGMPC

You may read the original readme with build and usage instructions as [`README-ORIGINAL.md`](README-ORIGINAL.md)
Also note, as this library depends on `emp-tool` and `emp-ot` for large parts of the cryptographic processing, optimizations there are also relevant here.

## Main Changes

|Keywords|File|Function|Summary|
|-|-|-|-|
|Preprocessing Batching|[`fpremp.h`](emp-agmpc/fpremp.h)|`FpreMP::garble()`, `FpreMP::evaluate()`, `FpreMP::HnIDProcess()`, `FpreMP::RecvProcess()`, `FpreMP::compute()`|Changed to batch several of these independent operations using a pre- and a post-processing loop around cryptographic operations.|
|Function-dependent and Online Batching|[`mpc.h`](emp-agmpc/mpc.h)|`CMPC::Hash()`, `CMPC::EvaluateANDGates()`, `CMPC::GarbleGates`, `CMPC::online()`, `CMPC::function_dependent()`|Batching for the actual function- and data-dependent operations.|