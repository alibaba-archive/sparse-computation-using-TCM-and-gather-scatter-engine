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
    parser.add_argument('--ni', type=int, required=True)
    parser.add_argument('--vl', type=int, required=True, help="vector length")
    parser.add_argument('--fp16', action="store_true")
    parser.add_argument('--fp32', action="store_true")
    args = parser.parse_args()
    return args


def write_dense_split(args):
    body = ["""
#include <cassert>

// #include <iostream>
#include "mm.h"

/*
  Assume matrix B is reshaped to (N_o, K, N_i)
  Loop order is M_o, N_o, K, M_i, N_i
  N_i is vectorized, M_i is unrolled
*/
void hgemm_unrolled_split(float16_t *C, const float16_t *A, const float16_t *B,
    unsigned long M, unsigned long K, unsigned long N) {{
  unsigned M_i = {};
  unsigned M_m = {};
  unsigned N_i = {};
  assert(svcnth() == {});
  assert((N_i % svcnth()) == 0);
  assert(M > (M_i * M_m));
  assert(M % (M_i * M_m) == 0);
  // int nn = (int)(N_i / svcnth());
  const svbool_t all_active = svptrue_b16();

  for (unsigned long i = 0; i < M; i+= (M_i * M_m)) {{
    // std::cout << "i : " << i << std::endl;
    for (unsigned long j = 0; j < N; j += N_i) {{
      // std::cout << "j : " << j << std::endl;
      for (unsigned long l = 0; l < (M_i * M_m); l+= M_i) {{""".format(
      args.mi, args.mm, args.ni, args.vl)]
    assert(args.ni % args.vl == 0)
    nn = args.ni // args.vl

    for i in range(args.mi):
        for j in range(nn):
            body.append(
"""          svfloat16_t acc_m{}_n{} = svdup_f16(0);""".format(i, j))
    body.append(
"""      for (unsigned long k = 0; k < K; ++k) {""")
    for i in range(args.mi):
        body.append(
"""        const svfloat16_t A_{} = svdup_f16(A[(i + l + {}) * K + k]);""".\
        format(i, i))

    for j in range(nn):
        body.append("""
        const svfloat16_t B_k_{} = svld1_vnum(all_active, &B[k * N_i], {});"""\
        .format(j, j))

    for i in range(args.mi):
        for j in range(nn):
            body.append(
"""        acc_m{}_n{} = svmla_x(all_active, acc_m{}_n{}, A_{}, B_k_{});"""\
            .format(i, j, i, j, i, j))

    body.append(
"""      }""")
    for i in range(args.mi):
        for j in range(nn):
            body.append(
"""      svst1_vnum(all_active, &C[(i + l + {}) * N + {}], 0, acc_m{}_n{});"""\
            .format(i, j, i, j))

    body.append(
"""      }
      }
  }
}""")
    for i in range(len(body)):
        print(body[i])


def write_dense_vector_split(args):
    assert(args.ni % args.vl == 0)
    body = ["""
#include <cassert>

// #include <iostream>
#include "mm.h"

/*
  Matrix B is (No, K, Ni)
*/
void spmv_sve_split(float32_t *C, const float32_t *A, const float16_t *B,
              unsigned long K, unsigned long N) {{
  assert (N % svcnth() == 0);
  assert ({} % (svcnth()) == 0);
  assert (N % {} == 0);
  assert (svcnth() == 8);
  const svbool_t pred16lo = svdupq_b16(false, false, false, false,
                                       true, true, true, true);
  const svbool_t pred16hi = svdupq_b16(true, true, true, true,
                                       false, false, false, false);
  const svbool_t pred16 = svptrue_b16();
  const svbool_t pred32 = svptrue_b32();
""".format(args.ni, args.ni)]
    body.append("""
  for (unsigned i = 0; i < N; i += {}) {{
""".format(args.ni))
    for i in range(0, args.ni, args.vl):
        body.append("""
    svfloat32_t acc{}_hi = svdup_f32(0);
    svfloat32_t acc{}_lo = svdup_f32(0);
    """.format(i, i, i, i))
    body.append("""
    for (unsigned j = 0; j < K; j+=svcntw()) {
      const svfloat32_t A_v = svld1(pred32, &A[j]);""")
    for j in range (4):
        for i in range (0, args.ni, args.vl):
            body.append("""
      const svfloat16_t B{}_{} = svld1(pred16, &B[i * K + (j + {}) * {} + {}]);
      const svfloat32_t B{}_{}_hi = svcvt_f32_x(pred16hi, B{}_{});
      const svfloat32_t B{}_{}_lo = svcvt_f32_x(pred16lo, B{}_{});
      """.format(i, j, j, args.ni, i,
                 i, j, i, j,
                 i, j, i, j))
    for j in range (4):
        for i in range (0, args.ni, args.vl):
            body.append("""
      acc{}_hi = svmla_lane(acc{}_hi, B{}_{}_hi, A_v, {});
      acc{}_lo = svmla_lane(acc{}_lo, B{}_{}_lo, A_v, {});
      """.format(i, i, i, j, j,
                 i, i, i, j, j))

    body.append("""
    }""")
    for i in range(0, args.ni, args.vl):
        body.append("""
    svst1(pred32, &C[i + {}], acc{}_lo);
    svst1(pred32, &C[i + {} + svcntw()], acc{}_hi);
    """.format(i, i, i, i))
    body.append(
"""
  }
}""")
    for i in range(len(body)):
        print(body[i])


def write_dense_vector_split_fp16(args):
    assert(args.ni % args.vl == 0)
    body = ["""
#include <cassert>

// #include <iostream>
#include "mm.h"

/*
  Matrix B is (No, K, Ni)
*/
void mm_sve_split_fp16(float16_t *C, const float16_t *A, const float16_t *B,
              unsigned long K, unsigned long N) {{
  assert (N % svcnth() == 0);
  assert ({} % (svcnth()) == 0);
  assert (N % {} == 0);
  assert (svcnth() == 8);

  const svbool_t pred16 = svptrue_b16();
""".format(args.ni, args.ni)]
    body.append("""
  for (unsigned i = 0; i < N; i += {}) {{
""".format(args.ni))
    for i in range(0, args.ni):
        body.append("""    svfloat16_t acc{} = svdup_f16(0);""".format(i))
    body.append("""
    for (unsigned j = 0; j < K; j+=svcnth()) {
      const svfloat16_t A_v = svld1(pred16, &A[j]);""")
    for i in range (0, args.ni):
        body.append("""
      const svfloat16_t B{} = svld1(pred16, &B[(i+{}) * K + j]);
      acc{} = svmla_x(pred16, acc{}, B{}, A_v);""".format(i, i, i, i, i))

    body.append("""
    }""")
    for i in range(0, args.ni):
        body.append("""    float16_t a{} = svaddv(pred16, acc{});
    C[i+{}] = a{};""".format(i, i, i, i))
    body.append(
"""
  }
}""")

def write_dense_vector_split_fp32(args):
    assert(args.ni % args.vl == 0)
    body = ["""
#include <cassert>

// #include <iostream>
#include "mm.h"

/*
  Matrix B is (No, K, Ni)
*/
void mm_sve_split_fp16(float16_t *C, const float16_t *A, const float16_t *B,
              unsigned long K, unsigned long N) {{
  assert (N % svcnth() == 0);
  assert ({} % (svcnth()) == 0);
  assert (N % {} == 0);
  assert (svcnth() == 8);

  const svbool_t pred16 = svptrue_b16();
  const svbool_t pred32 = svptrue_b32();
  const svbool_t pred16lo = svdupq_b16(false, false, false, false,
                                          true, true, true, true);
  const svbool_t pred16hi = svdupq_b16(true, true, true, true,
                                          false, false, false, false);
""".format(args.ni, args.ni)]
    body.append("""
  for (unsigned i = 0; i < N; i += {}) {{
""".format(args.ni))
    for i in range(0, args.ni*2):
        body.append("""    svfloat32_t acc{} = svdup_f32(0);""".format(i))
    body.append("""
    for (unsigned j = 0; j < K; j+=svcnth()) {
      const svfloat16_t A_v = svld1(pred16, &A[j]);
      const svfloat32_t A_v_hi = svcvt_f32_x(pred16hi, A_v);
      const svfloat32_t A_v_lo = svcvt_f32_x(pred16lo, A_v);""")
    for i in range (0, args.ni):
        body.append("""
      const svfloat16_t B{} = svld1(pred16, &B[(i+{}) * K + j]);
      const svfloat32_t B{}_hi = svcvt_f32_x(pred16hi, B{});
      const svfloat32_t B{}_lo = svcvt_f32_x(pred16lo, B{});
      acc{} = svmla_x(pred32, acc{}, B{}_hi, A_v_hi);
      acc{} = svmla_x(pred32, acc{}, B{}_lo, A_v_lo);""".format(i, i, i, i, i, i, i*2, i*2, i, i*2+1, i*2+1 ,i))

    body.append("""
    }""")
    for i in range(0, args.ni):
        body.append("""    acc{} = svadd_z(pred32, acc{}, acc{});
    float32_t a{} = svaddv(pred32, acc{});
    C[i+{}] = a{};""".format(i*2, i*2, i*2+1, i, i*2, i, i))
    body.append(
"""
  }
}""")
    for i in range(len(body)):
        print(body[i])

if __name__ == "__main__":
    args = parse()
    if args.m == 1:
        print("#if DENSE_FP16")
        print("#if COMPUTE_FP32")
        write_dense_vector_split_fp32(args)
        print("#else // COMPUTE_FP32")
        write_dense_vector_split_fp16(args)
        print("#endif // COMPUTE_FP32")
        print("#else // DENSE_FP16")
        write_dense_vector_split(args)
        print("#endif // DENSE_FP16")
    else:
            write_dense_split(args)
