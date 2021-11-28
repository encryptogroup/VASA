# EMP-OT

You may read the original readme with build and usage instructions as [`README-ORIGINAL.md`](README-ORIGINAL.md).
Also note, as this library depends on `emp-tool` for large parts of the cryptographic processing, optimizations there are also relevant here.

## Main Changes

|Keywords|File|Function|Summary|
|-|-|-|-|
|LPN Processing Widening|[`lpn_f2.h`](emp-ot/ferret/lpn_f2.h)|`LpnF2::__compute4()`|Increased batch size for use of VAES instead of AES-NI.|
|VAES for GGM Processing|[`twokeyprp.h`](emp-ot/ferret/twokeyprp.h)|`TwoKeyPRP::node_expand_double_vaes()`, `TwoKeyPRP::node_expand_double()`|Generalized the PRG implementation and added VAES for processing if requested.|


