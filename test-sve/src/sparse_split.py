
'''
/*
 * Copyright (c) 2020
 * All rights reserved
 *
 * The license below extends only to copyright in the software and shall
 * not be construed as granting a license to any other intellectual
 * property including but not limited to intellectual property relating
 * to a hardware implementation of the functionality of the software
 * licensed hereunder.  You may use the software subject to the license
 * terms below provided that you ensure that this notice is replicated
 * unmodified and in its entirety in all distributions of the software,
 * modified or unmodified, in source code or in binary form.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met: redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer;
 * redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution;
 * neither the name of the copyright holders nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */
 '''
import argparse

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--m', type=int, required=True)
    parser.add_argument('--mi', type=int, required=True)
    parser.add_argument('--mm', type=int, required=True)
    parser.add_argument('--k', type=int, required=True)
    parser.add_argument('--n', type=int, required=True)
    # parser.add_argument('--ni', type=int, required=True)
    parser.add_argument('--vl', type=int, required=True, help="vector length")
    parser.add_argument('--trans', action='store_true')
    args = parser.parse_args()
    return args


def write_sparse_split(args):
    transposed = args.trans

    body = ["""
#include <cassert>

// #include <iostream>
#include "mm.h"

/*
  Matrix A is {}
  Block sparse format, the B_indptr saves the block number
  i.e. the max value is N / BLOCK. N must be divisable of BLOCK
  The tiling is not done
*/
void spmm_block_vertical_split{}(float16_t *C, const float16_t *A,
    const float16_t *B_value, const unsigned short *B_indices,
    const unsigned int *B_indptr,
    unsigned long M, unsigned long K,
    unsigned long N, unsigned block) {{
  unsigned M_i = {};
  unsigned M_m = {};
  assert(svcnth() == {});
  // assert((N_i % svcnth()) == 0);
  assert(M >= (M_i * M_m));
  assert(M % (M_i * M_m) == 0);
  assert(block == svcnth());
  const svbool_t pred = svptrue_b16();
  for (unsigned long i = 0; i < M; i+= (M_m * M_i)) {{
    // std::cout << "i : " << i << std::endl;
    for (unsigned int j = 0; j < (unsigned)(N / block); j++) {{
      // std::cout << "j: " << j << std::endl;
      unsigned int start = B_indptr[j];
      unsigned int end = B_indptr[j+1];
      for (unsigned long l = 0; l < (M_i * M_m); l+= M_i) {{""".format(
      "transposed: (K, M), assumes M is divisible by svcnth()"
      if transposed else "(M, K), ",
      "_transposed" if transposed else "",
      args.mi, args.mm, args.vl)]

    for i in range(args.mi):
        body.append(
"""          svfloat16_t acc_m{} = svdup_f16(0);""".format(i))
    body.append(
"""      for (unsigned long k = start; k < end; ++k) {
            const svfloat16_t B_v = svld1(pred, &B_value[k * block]);
            unsigned short B_k = B_indices[k];""")

    if transposed:
        for i in range(0, args.mi, args.vl):
            body.append(
"""            const svfloat16_t A_{} = svld1_vnum(pred,
                    &A[B_k * M], ((int)(i + l)/{} + {}));""".\
                    format(i, args.vl, i))
            for j in range(args.mi):
                body.append(
"""            acc_m{} = svmla_lane(acc_m{}, A_{}, B_v, {});"""
                .format(i * args.vl + j, i *args.vl + j, i, j))
    else:
        for i in range(args.mi):
            body.append(
"""            const svfloat16_t A_{} = svdup_f16(
                    A[(i + l + {}) * K + B_k]);""".format(i, i))

        for i in range(args.mi):
            body.append(
"""            acc_m{} = svmla_x(pred, acc_m{}, A_{}, B_v);"""\
            .format(i, i, i))

    body.append(
"""      }""")
    for i in range(args.mi):
        body.append(
"""      svst1_vnum(pred, &C[(i + l + {}) * N], j, acc_m{});"""\
            .format(i, i))

    body.append(
"""      }
      }
  }
}""")
    for i in range(len(body)):
        print(body[i])


if __name__ == "__main__":
    args = parse()
    write_sparse_split(args)
