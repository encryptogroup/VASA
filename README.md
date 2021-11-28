# Vector AES Instructions for Security Applications

This is the supporting code repository for the paper "VASA: Vector AES Instructions for Security Applications" to appear and be presented at [ACSAC'21](https://www.acsac.org/) by Jean-Pierre Münch, Hossein Yalame, and Thomas Schneider.

The intention behind this code and repository is primarily to show how to use VAES and give concrete examples of how we used it, so you can adapt it to your library / framework / application.

The suggested way to work with this repository is as follows:

1. Identify the framework you're interested in, i.e. CrypTFlow2, ABY, EMP-OT (relies on EMP-Tool), EMP-AG2PC (relies on EMP-Tool and EMP-OT), or EMP-AGMPC (relies on EMP-Tool and EMP-OT)
2. Read the Readmes in the relevant folders.
3. Navigate to the files / functions that are designated in these readmes for the changes done to the code.
4. Read the new code and its surrounding documentation.
5. Optionally, compare this to the simpler baseline implementation to get a more iterative understanding of the changes.

## Repository Organization

|Name|Path|Readme|Baseline Commit|License|Copyright Holder|
|-|-|-|-|-|-|
|ABY|`ABY/`|[`ABY/README.md`](ABY/README.md)|[`08baa853de76a9070cb8ed8d41e96569776e4773`](https://github.com/encryptogroup/ABY/tree/08baa853de76a9070cb8ed8d41e96569776e4773)|[LGPLv3](ABY/LICENSE)|ENCRYPTO|
|CrypTFlow2 / SCI|`SCI/`|[`SCI/README.md`](SCI/README.md)|[`3f72d1519529279a47d9c2bc01799d7e65db07e1`](https://github.com/mpc-msri/EzPC/tree/3f72d1519529279a47d9c2bc01799d7e65db07e1)|[MIT](SCI/LICENSE)|Microsoft Research|
|EMP-Tool|`emp/emp-tool/`|[`emp/emp-tool/README.md`](emp/emp-tool/README.md)|[`ef7a54564d30a4243ee710e0df79323c94f5c9f9`](https://github.com/emp-toolkit/emp-tool/tree/ef7a54564d30a4243ee710e0df79323c94f5c9f9)|[MIT](emp/emp-tool/LICENSE)|Xiao Wang|
|EMP-OT|`emp/emp-ot/`|[`emp/emp-ot/README.md`](emp/emp-ot/README.md)|[`f5aa97337b7f30cbf8ccbb4a763860e6576a8108`](https://github.com/emp-toolkit/emp-ot/tree/f5aa97337b7f30cbf8ccbb4a763860e6576a8108)|[MIT](emp/emp-ot/LICENSE)|Xiao Wang|
|EMP-AG2PC|`emp/emp-ag2pc/`|[`emp/emp-ag2pc/README.md`](emp/emp-ag2pc/README.md)|[`11e51179a2a5e09ba8e7f3736ae955966b96fc92`](https://github.com/emp-toolkit/emp-ag2pc/tree/11e51179a2a5e09ba8e7f3736ae955966b96fc92)|[MIT](emp/emp-ag2pc/LICENSE)|Xiao Wang|
|EMP-AGMPC|`emp/emp-agmpc/`|[`emp/emp-agmpc/README.md`](emp/emp-agmpc/README.md)|[`7d30b53630e2b25469811ab014e4d4a26697a89c`](https://github.com/emp-toolkit/emp-agmpc/tree/7d30b53630e2b25469811ab014e4d4a26697a89c)|[MIT](emp/emp-agmpc/LICENSE)|Xiao Wang|

All code changes are licensed under the same license as the library originally had.
