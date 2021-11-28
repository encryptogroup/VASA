/*
Authors: Deevashwer Rathee
Copyright:
Copyright (c) 2020 Microsoft Research
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#ifndef CCRF_H__
#define CCRF_H__
#include "utils/prg.h"
#include "utils/aes-ni.h"
#include "utils/aes_opt.h"
#include "utils/vaes.h"
#include <stdio.h>
/** @addtogroup BP
  @{
  */
namespace sci {

	inline void CCRF(block128* y, block256* k, int n) {
		AESNI_KEY aes[8];
        int r = n % 8;

		int vaes_remainder = n % 16;
		int vaes_bulk = n - vaes_remainder;
		int aesni_remainder = vaes_remainder % 8;
		int aesni_bulk = vaes_remainder - aesni_remainder;

		int i = 0;

		if (vaes_bulk > 0)
		{
			for (; i < vaes_bulk; i += 16)
				VAES256_KS_ENC_ONE<16>(y+i, reinterpret_cast<const block128*>(k+i));
		}
		if (aesni_bulk > 0)
		{
			for (; i < vaes_bulk + aesni_bulk; i+=8) {
				AES_256_ks8(k + i, aes);
				//AESNI_set_encrypt_key(&aes[j], k[i*8 + j]);
				AESNI_ecb_encrypt_blks_ks_x8(y + i, 8, aes);
			}
		}
		if (aesni_remainder > 0)
		{
			for (; i < n; i++) {
				y[i] = one;
				AESNI_set_encrypt_key(&aes[0], k[i]);
				AESNI_ecb_encrypt_blks(y + i, 1, aes);
			}
		}
	}

}
/**@}*/
#endif// CCRF_H__
