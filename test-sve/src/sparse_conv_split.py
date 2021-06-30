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
 * Author: Haoran Li
 */
'''
import argparse

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sp_unroll', type=int, required=True)
    parser.add_argument('--vl', type=int, required=True, help="vector length")
    args = parser.parse_args()
    return args


def write_sparse_horizontal_split_loop(vl, unroll):
    body = [
"""    while (idx + {} <= end ) {{""".
    format(unroll)]
    for i in range(0, unroll):
        body.append(
"""      svuint16_t W{}_k = svld1_vnum_u16(pred16, &W_join[idx * step], {});
      svuint16_t W{}_v_u = svld1_vnum_u16(pred16, &W_join[idx * step], {});
      svfloat16_t W{}_v = svreinterpret_f16(W{}_v_u);
      svfloat16_t W{}_v_odd16 = svtrn2(W{}_v, W{}_v);
      const svfloat32_t W{}_v_even = svcvt_f32_z(pred16, W{}_v);
      const svfloat32_t W{}_v_odd = svcvt_f32_z(pred16, W{}_v_odd16);""".
      format(i, i * 2 + 1, i, i * 2, i, i, i, i, i, i, i, i, i))
    body.append("""
      for (unsigned int h = 0; h < H_out; h++) {
        for (unsigned int w = 0; w < W_out; w++) {
          svfloat32_t acc = svdup_f32(0);
          svfloat32_t acc1 = svdup_f32(0);
          float16_t const * A_addr = &A_padded[(h * A_w + w) * C_in];
          unsigned oidx = (h * W_out + w) * C_out + co; // NHWC""")
    for i in range(0, unroll):
        body.append("""
          svfloat16_t A{}_g = svreinterpret_f16(
            svld1_gather_index(pred16, (float64_t *)A_addr,
            svreinterpret_u64(W{}_k)));
          svfloat16_t A{}_g_odd16 = svtrn2(A{}_g, A{}_g);
          const svfloat32_t A{}_g_even = svcvt_f32_z(pred16, A{}_g);
          const svfloat32_t A{}_g_odd = svcvt_f32_z(pred16, A{}_g_odd16);    	            
          acc = svmla_x(pred32, acc, A{}_g_even, W{}_v_even); //perform mla
          acc1 = svmla_x(pred32, acc1, A{}_g_odd, W{}_v_odd); //perform mla""".
          format(i, i, i, i, i, i, i, i, i, i, i, i, i))
    body.append("""
          acc = svadd_z(pred32, acc, acc1);
          float32_t r = svaddv(pred32, acc);
          C[oidx] = r;
          num += 8;
        }}
      }}
      idx = idx + {};
    }}""".format(unroll))
    return body


def write_sparse_horizontal_split(args):

    body = ["""
#include <cassert>
#include <iostream>

#include "mm.h"

void sp_conv_horizontal_fp16_fp32_compute(
                  float16_t *C, float16_t const *A_padded,
                  unsigned short const *W_join,
                  const unsigned int *W_indptr,
                  int A_h, int A_w, int W_h, int W_w, int C_in,
                  unsigned int C_out,
                  int nitems) {{
  assert(nitems == {});
  assert({} == 8 or {} == 16 or {} == 12);
  unsigned int H_out = A_h - W_h + 1;
  unsigned int W_out = A_w - W_w + 1;
  // const svbool_t pred16lo = svdupq_b16(false, false, false, false,
  //                                      true, true, true, true);
  // const svbool_t pred16hi = svdupq_b16(true, true, true, true,
  //                                      false, false, false, false);
  const svbool_t pred16 = svptrue_b16();
  const svbool_t pred32 = svptrue_b32();
  int num = 0;
  for (unsigned int co = 0; co < C_out; co++) {{
    unsigned int start = W_indptr[co];
    unsigned int end = W_indptr[co+1];
    //std::cout << start << " : " << end << std::endl;
    unsigned idx = start;
    unsigned step = 2 * nitems;""".
    format(args.vl, args.sp_unroll, args.sp_unroll, args.sp_unroll)]
    unroll = args.sp_unroll
    while unroll > 0:
        body = body + write_sparse_horizontal_split_loop(args.vl, unroll)
        unroll = unroll // 2
    body.append("""
  }
}""")

    for i in range(len(body)):
        print(body[i])

def write_sparse_vertical_split_loop(vl, unroll):
    body = [
    """    while (idx + {} <= end ) {{""".
    format(unroll)]
    for i in range(0, unroll):
        body.append(
"""      svuint16_t W{}_k = svld1_vnum_u16(pred16, &W_join[idx * step], {});
      svuint16_t W{}_v_u = svld1_vnum_u16(pred16, &W_join[idx * step], {});
      svfloat16_t W{}_v = svreinterpret_f16(W{}_v_u);
      svfloat16_t W{}_v_odd16 = svtrn2(W{}_v, W{}_v);
      const svfloat32_t W{}_v_even = svcvt_f32_z(pred16, W{}_v);
      const svfloat32_t W{}_v_odd = svcvt_f32_z(pred16, W{}_v_odd16);""".
      format(i, i * 2 + 1, i, i * 2, i, i, i, i, i, i, i, i, i))
    body.append("""
      for (unsigned int h = 0; h < H_out; h++) {
        for (unsigned int w = 0; w < W_out; w++) {
        //std::cout << idx << " : " << h << " : " << w << std::endl;
          svfloat32_t acc = svdup_f32(0);
          svfloat32_t acc1 = svdup_f32(0);
          float16_t const * A_addr = &A_padded[(h * A_w + w) * C_in];
          unsigned oidx = (h * W_out + w) * C_out + co*nitems; // NHWC""")
    for i in range(0, unroll):
        body.append("""
          svfloat16_t A{}_g = svreinterpret_f16(
            svld1_gather_index(pred16, (float64_t *)A_addr,
            svreinterpret_u64(W{}_k)));
          svfloat16_t A{}_g_odd16 = svtrn2(A{}_g, A{}_g);
          const svfloat32_t A{}_g_even = svcvt_f32_x(pred16, A{}_g);
          const svfloat32_t A{}_g_odd = svcvt_f32_x(pred16, A{}_g_odd16);
          acc = svmla_x(pred32, acc, A{}_g_even, W{}_v_even); //perform mla
          acc1 = svmla_x(pred32, acc1, A{}_g_odd, W{}_v_odd); //perform mla""".
          format(i, i, i, i, i, i, i, i, i, i, i, i, i))
    body.append("""
          svfloat16_t acc16_odd = svcvt_f16_z(pred16, acc1);
          svfloat16_t acc16_even = svcvt_f16_z(pred16, acc);
          svfloat16_t acc16 = svtrn1(acc16_even, acc16_odd);

          svfloat16_t c = svld1_f16(pred16, &C[oidx]);
          acc16 = svadd_f16_z(pred16, acc16, c);
          svst1(pred16, &C[oidx], acc16);
        }}
      }}
      idx = idx + {};
    }}""".format(unroll))
    return body


def write_sparse_vertical_split(args):

    body = ["""
#include <cassert>
#include <iostream>

#include "mm.h"

void sp_conv_vertical_fp16_fp32_compute(
                  float16_t *C, float16_t const *A_padded,
                  unsigned short const *W_join,
                  const unsigned int *W_indptr,
                  int A_h, int A_w, int W_h, int W_w, int C_in,
                  unsigned int C_out,
                  int nitems) {{
  assert(nitems == {});
  assert({} == 8 or {} == 16 or {} == 12);
  unsigned int H_out = A_h - W_h + 1;
  unsigned int W_out = A_w - W_w + 1;
  // const svbool_t pred16lo = svdupq_b16(false, false, false, false,
  //                                      true, true, true, true);
  // const svbool_t pred16hi = svdupq_b16(true, true, true, true,
  //                                      false, false, false, false);
  const svbool_t pred16 = svptrue_b16();
  const svbool_t pred32 = svptrue_b32();
  for (unsigned int co = 0; co < C_out/nitems; co++) {{

    unsigned int start = W_indptr[co];
    unsigned int end = W_indptr[co+1];
    //std::cout << start << " : " << end << std::endl;
    unsigned idx = start;
    unsigned step = 2 * nitems;""".
    format(args.vl, args.sp_unroll, args.sp_unroll, args.sp_unroll)]
    unroll = args.sp_unroll
    while unroll > 0:
        body = body + write_sparse_vertical_split_loop(args.vl, unroll)
        unroll = unroll // 2
    body.append("""
  }
}""")

    for i in range(len(body)):
        print(body[i])


def write_sparse_block_vertical_split_loop(vl, unroll):
    body = [
    """    while (idx + {} <= end ) {{""".
    format(unroll)]
    for i in range(0, unroll):
        body.append(
"""   svfloat16_t W{}_v = svld1_vnum(pred16, &W[(idx) * nitems], {});
      svfloat16_t W{}_v_odd16 = svtrn2(W{}_v, W{}_v);
      const svfloat32_t W{}_v_even = svcvt_f32_z(pred16, W{}_v);
      const svfloat32_t W{}_v_odd = svcvt_f32_z(pred16, W{}_v_odd16);""".
      format(i, i, i, i, i, i, i, i, i))
    body.append("""
      for (unsigned int h = 0; h < H_out; h++) {
        for (unsigned int w = 0; w < W_out; w++) {
        //std::cout << idx << " : " << h << " : " << w << std::endl;
          svfloat32_t acc = svdup_f32(0);
          svfloat32_t acc1 = svdup_f32(0);
          float16_t const * A_addr = &A_padded[(h * A_w + w) * C_in];
          unsigned oidx = (h * W_out + w) * C_out + co*nitems; // NHWC""")
    for i in range(0, unroll):
        body.append("""
          unsigned short A{}_g_K = W_indices[idx+{}];
          const svfloat32_t A{}_g = svdup_f32(A_addr[A{}_g_K]);
          acc = svmla_x(pred32, acc, A{}_g, W{}_v_even); //perform mla
          acc1 = svmla_x(pred32, acc1, A{}_g, W{}_v_odd); //perform mla""".
          format(i, i, i, i, i, i, i, i))
    body.append("""
          svfloat16_t acc16_odd = svcvt_f16_z(pred16, acc1);
          svfloat16_t acc16_even = svcvt_f16_z(pred16, acc);
          svfloat16_t acc16 = svtrn1(acc16_even, acc16_odd);

          svfloat16_t c = svld1_f16(pred16, &C[oidx]);
          acc16 = svadd_f16_z(pred16, acc16, c);
          svst1(pred16, &C[oidx], acc16);
        }}
      }}
      idx = idx + {};
    }}""".format(unroll))
    return body


def write_sparse_block_vertical_split(args):

    body = ["""
#include <cassert>
#include <iostream>

#include "mm.h"

void sp_conv_block_vertical_fp16_fp32_compute(
                  float16_t *C, float16_t const *A_padded,
                  const float16_t *W,
                  const unsigned short *W_indices,
                  const unsigned int *W_indptr,
                  int A_h, int A_w, int W_h, int W_w, int C_in,
                  unsigned int C_out,
                  int nitems) {{
  assert(nitems == {});
  assert({} == 8 or {} == 16 or {} == 12);
  unsigned int H_out = A_h - W_h + 1;
  unsigned int W_out = A_w - W_w + 1;
  const svbool_t pred16 = svptrue_b16();
  const svbool_t pred32 = svptrue_b32();
  for (unsigned int co = 0; co < C_out/nitems; co++) {{
    unsigned int start = W_indptr[co];
    unsigned int end = W_indptr[co+1];
    //std::cout << start << " : " << end << std::endl;
    unsigned idx = start;
    unsigned step = 2 * nitems;""".
    format(args.vl, args.sp_unroll, args.sp_unroll, args.sp_unroll)]
    unroll = args.sp_unroll
    while unroll > 0:
        body = body + write_sparse_block_vertical_split_loop(args.vl, unroll)
        unroll = unroll // 2
    body.append("""
  }
}""")

    for i in range(len(body)):
        print(body[i])


def write_sparse_block_horizontal_split_loop(vl, unroll):
    body = [
    """    while (idx + {} <= end ) {{""".
    format(unroll)]
    for i in range(0, unroll):
        body.append(
"""   svfloat16_t W{}_v = svld1_vnum(pred16, &W[(idx) * nitems], {});
      svfloat16_t W{}_v_odd16 = svtrn2(W{}_v, W{}_v);
      const svfloat32_t W{}_v_even = svcvt_f32_z(pred16, W{}_v);
      const svfloat32_t W{}_v_odd = svcvt_f32_z(pred16, W{}_v_odd16);""".
      format( i, i, i, i, i, i, i, i, i))
    body.append("""
      for (unsigned int h = 0; h < H_out; h++) {
        for (unsigned int w = 0; w < W_out; w++) {
          //std::cout << idx << " : " << h << " : " << w << std::endl;
          svfloat32_t acc = svdup_f32(0);
          svfloat32_t acc1 = svdup_f32(0);
          float16_t const * A_addr = &A_padded[(h * A_w + w) * C_in];
          unsigned oidx = (h * W_out + w) * C_out + co; // NHWC""")
    for i in range(0, unroll):
        body.append("""
          unsigned short A{}_g_K = W_indices[idx +{}];
          const svfloat16_t A{}_g = svld1_f16(pred16,
                (const float16_t *)&A_addr[A{}_g_K * nitems]);
          svfloat16_t   A{}_g_odd16 = svtrn2(A{}_g, A{}_g);
          const svfloat32_t A{}_g_even = svcvt_f32_x(pred16, A{}_g);
          const svfloat32_t A{}_g_odd = svcvt_f32_x(pred16, A{}_g_odd16);
          acc = svmla_x(pred32, acc, A{}_g_even, W{}_v_even); //perform mla
          acc1 = svmla_x(pred32, acc1, A{}_g_odd, W{}_v_odd); //perform mla""".
          format(i, i, i, i, i, i, i, i, i, i, i, i, i, i, i))
    body.append("""
          svfloat16_t acc16_odd = svcvt_f16_z(pred16, acc1);
          svfloat16_t acc16_even = svcvt_f16_z(pred16, acc);
          svfloat16_t acc16 = svtrn1(acc16_even, acc16_odd);
          float32_t r = svaddv(pred16, acc16);
          C[oidx] = r;
        }}
      }}
      idx = idx + {};
    }}""".format(unroll))
    return body


def write_sparse_block_horizontal_split(args):

    body = ["""
#include <cassert>
#include <iostream>

#include "mm.h"

void sp_conv_block_horizontal_fp16_fp32_compute(
                  float16_t *C, float16_t const *A_padded,
                  const float16_t *W,
                  const unsigned short *W_indices,
                  const unsigned int *W_indptr,
                  int A_h, int A_w, int W_h, int W_w, int C_in,
                  unsigned int C_out,
                  int nitems) {{
  assert(nitems == {});
  assert({} == 8 or {} == 16 or {} == 12);
  unsigned int H_out = A_h - W_h + 1;
  unsigned int W_out = A_w - W_w + 1;
  // const svbool_t pred16lo = svdupq_b16(false, false, false, false,
  //                                      true, true, true, true);
  // const svbool_t pred16hi = svdupq_b16(true, true, true, true,
  //                                      false, false, false, false);
  const svbool_t pred16 = svptrue_b16();
  const svbool_t pred32 = svptrue_b32();
  for (unsigned int co = 0; co < C_out; co++) {{
    unsigned int start = W_indptr[co];
    unsigned int end = W_indptr[co+1];
    //std::cout << start << " : " << end << std::endl;
    unsigned idx = start;
    unsigned step = 2 * nitems;""".
    format(args.vl, args.sp_unroll, args.sp_unroll, args.sp_unroll)]
    unroll = args.sp_unroll
    while unroll > 0:
        body = body + write_sparse_block_horizontal_split_loop(args.vl, unroll)
        unroll = unroll // 2
    body.append("""
  }
}""")

    for i in range(len(body)):
        print(body[i])


def write_sparse_horizontal_split_fp16_loop(vl, unroll):
    body = [
"""    while (idx + {} <= end ) {{""".
    format(unroll)]
    for i in range(0, unroll):
        body.append(
"""      svuint16_t W{}_k = svld1_vnum_u16(pred16, &W_join[idx * step], {});
      svuint16_t W{}_v_u = svld1_vnum_u16(pred16, &W_join[idx * step], {});
      svfloat16_t W{}_v = svreinterpret_f16(W{}_v_u);""".
      format(i, i * 2 + 1, i, i * 2, i, i))
    body.append("""
      for (unsigned int h = 0; h < H_out; h++) {
        for (unsigned int w = 0; w < W_out; w++) {
          svfloat16_t acc = svdup_f16(0);
          float16_t const * A_addr = &A_padded[(h * A_w + w) * C_in];
          unsigned oidx = (h * W_out + w) * C_out + co; // NHWC""")
    for i in range(0, unroll):
        body.append("""
          svfloat16_t A{}_g = svreinterpret_f16(
            svld1_gather_index(pred16, (float64_t *)A_addr,
            svreinterpret_u64(W{}_k)));    	            
          acc = svmla_x(pred16, acc, A{}_g, W{}_v); //perform mla""".
          format(i, i, i, i))
    body.append("""
          float16_t r = svaddv(pred16, acc);
          C[oidx] = r;
          num += 8;
        //   std::cout << co << " " << h << " " << w << " " << std::endl;
        }}
      }}
      //   std::cout << co  << std::endl;
      idx = idx + {};
    }}""".format(unroll))
    return body


def write_sparse_horizontal_split_fp16(args):

    body = ["""
#include <cassert>
#include <iostream>

#include "mm.h"

void sp_conv_horizontal_fp16(
                  float16_t *C, float16_t const *A_padded,
                  unsigned short const *W_join,
                  const unsigned int *W_indptr,
                  int A_h, int A_w, int W_h, int W_w, int C_in,
                  unsigned int C_out,
                  int nitems) {{
  assert(nitems == {});
  assert({} == 8 or {} == 16 or {} == 12);
  unsigned int H_out = A_h - W_h + 1;
  unsigned int W_out = A_w - W_w + 1;
  const svbool_t pred16 = svptrue_b16();
  int num = 0;
  std::cout << " : " << std::endl;
  for (unsigned int co = 0; co < C_out; co++) {{
    unsigned int start = W_indptr[co];
    unsigned int end = W_indptr[co+1];
    //std::cout << start << " : " << end << std::endl;
    unsigned idx = start;
    unsigned step = 2 * nitems;""".
    format(args.vl, args.sp_unroll, args.sp_unroll, args.sp_unroll)]
    unroll = args.sp_unroll
    while unroll > 0:
        body = body + write_sparse_horizontal_split_fp16_loop(args.vl, unroll)
        unroll = unroll // 2
    body.append("""
  }
}""")

    for i in range(len(body)):
        print(body[i])

def write_sparse_vertical_split_fp16_loop(vl, unroll):
    body = [
    """    while (idx + {} <= end ) {{""".
    format(unroll)]
    for i in range(0, unroll):
        body.append(
"""      svuint16_t W{}_k = svld1_vnum_u16(pred16, &W_join[idx * step], {});
      svuint16_t W{}_v_u = svld1_vnum_u16(pred16, &W_join[idx * step], {});
      svfloat16_t W{}_v = svreinterpret_f16(W{}_v_u);""".
      format(i, i * 2 + 1, i, i * 2, i, i))
    body.append("""
      for (unsigned int h = 0; h < H_out; h++) {
        for (unsigned int w = 0; w < W_out; w++) {
        //std::cout << idx << " : " << h << " : " << w << std::endl;
          svfloat16_t acc = svdup_f16(0);
          float16_t const * A_addr = &A_padded[(h * A_w + w) * C_in];
          unsigned oidx = (h * W_out + w) * C_out + co*nitems; // NHWC""")
    for i in range(0, unroll):
        body.append("""
          svfloat16_t A{}_g = svreinterpret_f16(
            svld1_gather_index(pred16, (float64_t *)A_addr,
            svreinterpret_u64(W{}_k)));
          acc = svmla_x(pred16, acc, A{}_g, W{}_v); //perform mla""".
          format(i, i, i, i))
    body.append("""
          svfloat16_t c = svld1_f16(pred16, &C[oidx]);
          acc = svadd_f16_z(pred16, acc, c);
          svst1(pred16, &C[oidx], acc);
        }}
      }}
      idx = idx + {};
    }}""".format(unroll))
    return body

def write_sparse_vertical_split_fp16_fused_loop(vl, unroll):
    body = [
    """    while (idx + {} <= end ) {{
            auto idx_tmp = idx * step;
    """.
    format(unroll)]
    for i in range(0, unroll):
        body.append(
"""
      int W{}_ind = idx_tmp + nitems*{};
      svuint16_t W{}_v_u = svld1_vnum_u16(pred16, &W_join[idx_tmp], {});
      svfloat16_t W{}_v = svreinterpret_f16(W{}_v_u);""".
      format(i, i*2 + 1, i, i * 2, i, i))
    body.append("""
      for (unsigned int h = 0; h < H_out; h++) {
        for (unsigned int w = 0; w < W_out; w++) {
        //std::cout << idx << " : " << h << " : " << w << std::endl;
          svfloat16_t acc = svdup_f16(0);
          float16_t const * A_addr = &A_padded[(h * A_w + w) * C_in];
          unsigned oidx = (h * W_out + w) * C_out + co*nitems; // NHWC""")
    for i in range(0, unroll):
        body.append("""
          svfloat16_t A{}_g =
          spcpuFusedGather_f16(pred16, &W_join[W{}_ind], A_addr);
          acc = svmla_x(pred16, acc, A{}_g, W{}_v); //perform mla""".
          format(i, i, i, i))
    body.append("""
          svfloat16_t c = svld1_f16(pred16, &C[oidx]);
          acc = svadd_f16_z(pred16, acc, c);
          svst1(pred16, &C[oidx], acc);
        }}
      }}
      idx = idx + {};
    }}""".format(unroll))
    return body

def write_sparse_vertical_split_fp16(args):

    body = ["""
#include <cassert>
#include <iostream>

#include "mm.h"

void sp_conv_vertical_fp16(
                  float16_t *C, float16_t const *A_padded,
                  unsigned short const *W_join,
                  const unsigned int *W_indptr,
                  int A_h, int A_w, int W_h, int W_w, int C_in,
                  unsigned int C_out,
                  int nitems) {{
  assert(nitems == {});
  assert({} == 8 or {} == 16 or {} == 12);
  unsigned int H_out = A_h - W_h + 1;
  unsigned int W_out = A_w - W_w + 1;
  const svbool_t pred16 = svptrue_b16();
  std::cout << " : " << std::endl;
  for (unsigned int co = 0; co < C_out/nitems; co++) {{

    unsigned int start = W_indptr[co];
    unsigned int end = W_indptr[co+1];
    //std::cout << start << " : " << end << std::endl;
    unsigned idx = start;
    unsigned step = 2 * nitems;""".
    format(args.vl, args.sp_unroll, args.sp_unroll, args.sp_unroll)]
    unroll = args.sp_unroll
    while unroll > 0:
        body = body + write_sparse_vertical_split_fp16_loop(args.vl, unroll)
        unroll = unroll // 2
    body.append("""
  }
}""")

    for i in range(len(body)):
        print(body[i])

def write_sparse_vertical_split_fp16_fused(args):

    body = ["""
#include <cassert>
#include <iostream>

#include "mm.h"

void sp_conv_vertical_fp16_fused(
                  float16_t *C, float16_t const *A_padded,
                  unsigned short const *W_join,
                  const unsigned int *W_indptr,
                  int A_h, int A_w, int W_h, int W_w, int C_in,
                  unsigned int C_out,
                  int nitems) {{
  assert(nitems == {});
  assert({} == 8 or {} == 16 or {} == 12);
  unsigned int H_out = A_h - W_h + 1;
  unsigned int W_out = A_w - W_w + 1;
  const svbool_t pred16 = svptrue_b16();
  for (unsigned int co = 0; co < C_out/nitems; co++) {{

    unsigned int start = W_indptr[co];
    unsigned int end = W_indptr[co+1];
    //std::cout << start << " : " << end << std::endl;
    unsigned idx = start;
    unsigned step = 2 * nitems;""".
    format(args.vl, args.sp_unroll, args.sp_unroll, args.sp_unroll)]
    unroll = args.sp_unroll
    while unroll > 0:
      body = body + \
        write_sparse_vertical_split_fp16_fused_loop(args.vl, unroll)
      unroll = unroll // 2
    body.append("""
  }
}""")

    for i in range(len(body)):
        print(body[i])

def write_sparse_block_vertical_split_fp16_loop(vl, unroll):
    body = [
    """    while (idx + {} <= end ) {{""".
    format(unroll)]
    for i in range(0, unroll):
        body.append(
"""   svfloat16_t W{}_v = svld1_vnum(pred16, &W[(idx) * nitems], {});""".
      format(i, i))
    body.append("""
      for (unsigned int h = 0; h < H_out; h++) {
        for (unsigned int w = 0; w < W_out; w++) {
        //std::cout << idx << " : " << h << " : " << w << std::endl;
          svfloat16_t acc = svdup_f16(0);
          float16_t const * A_addr = &A_padded[(h * A_w + w) * C_in];
          unsigned oidx = (h * W_out + w) * C_out + co*nitems; // NHWC""")
    for i in range(0, unroll):
        body.append("""
          unsigned short A{}_g_K = W_indices[idx+{}];
          const svfloat16_t A{}_g = svdup_f16(A_addr[A{}_g_K]);
          acc = svmla_x(pred16, acc, A{}_g, W{}_v); //perform mla""".
          format(i, i, i, i, i, i))
    body.append("""

          svfloat16_t c = svld1_f16(pred16, &C[oidx]);
          acc = svadd_f16_z(pred16, acc, c);
          svst1(pred16, &C[oidx], acc);
        }}
      }}
      idx = idx + {};
    }}""".format(unroll))
    return body


def write_sparse_block_vertical_split_fp16(args):

    body = ["""
#include <cassert>
#include <iostream>

#include "mm.h"

void sp_conv_block_vertical_fp16(
                  float16_t *C, float16_t const *A_padded,
                  const float16_t *W,
                  const unsigned short *W_indices,
                  const unsigned int *W_indptr,
                  int A_h, int A_w, int W_h, int W_w, int C_in,
                  unsigned int C_out,
                  int nitems) {{
  assert(nitems == {});
  assert({} == 8 or {} == 16 or {} == 12);
  unsigned int H_out = A_h - W_h + 1;
  unsigned int W_out = A_w - W_w + 1;
  const svbool_t pred16 = svptrue_b16();
  std::cout << " : " << std::endl;
  for (unsigned int co = 0; co < C_out/nitems; co++) {{
    unsigned int start = W_indptr[co];
    unsigned int end = W_indptr[co+1];
    //std::cout << start << " : " << end << std::endl;
    unsigned idx = start;
    unsigned step = 2 * nitems;""".
    format(args.vl, args.sp_unroll, args.sp_unroll, args.sp_unroll)]
    unroll = args.sp_unroll
    while unroll > 0:
        body = body + write_sparse_block_vertical_split_fp16_loop(args.vl, unroll)
        unroll = unroll // 2
    body.append("""
  }
}""")

    for i in range(len(body)):
        print(body[i])


def write_sparse_block_horizontal_split_fp16_loop(vl, unroll):
    body = [
    """    while (idx + {} <= end ) {{""".
    format(unroll)]
    for i in range(0, unroll):
        body.append(
"""   svfloat16_t W{}_v = svld1_vnum(pred16, &W[(idx) * nitems], {});""".
      format( i, i))
    body.append("""
      for (unsigned int h = 0; h < H_out; h++) {
        for (unsigned int w = 0; w < W_out; w++) {
          //std::cout << idx << " : " << h << " : " << w << std::endl;
          svfloat16_t acc = svdup_f16(0);
          float16_t const * A_addr = &A_padded[(h * A_w + w) * C_in];
          //std::cout << A_padded[(h * A_w + w) * C_in] << std::endl;
          //std::cout << (h * A_w + w) * C_in << std::endl;
          unsigned oidx = (h * W_out + w) * C_out + co; // NHWC""")
    for i in range(0, unroll):
        body.append("""
          unsigned short A{}_g_K = W_indices[idx +{}];
          const svfloat16_t A{}_g = svld1_f16(pred16,
                (const float16_t *)&A_addr[A{}_g_K * nitems]);
          acc = svmla_x(pred16, acc, A{}_g, W{}_v); //perform mla""".
          format(i, i, i, i, i, i))
    body.append("""
          float16_t r = svaddv(pred16, acc);
          C[oidx] = r;
        }}
      }}
      idx = idx + {};
      //std::cout << idx << std::endl;
    }}""".format(unroll))
    return body


def write_sparse_block_horizontal_split_fp16(args):

    body = ["""
#include <cassert>
#include <iostream>

#include "mm.h"

void sp_conv_block_horizontal_fp16(
                  float16_t *C, float16_t const *A_padded,
                  const float16_t *W,
                  const unsigned short *W_indices,
                  const unsigned int *W_indptr,
                  int A_h, int A_w, int W_h, int W_w, int C_in,
                  unsigned int C_out,
                  int nitems) {{
  assert(nitems == {});
  assert({} == 8 or {} == 16 or {} == 12);
  unsigned int H_out = A_h - W_h + 1;
  unsigned int W_out = A_w - W_w + 1;
  const svbool_t pred16 = svptrue_b16();
  for (unsigned int co = 0; co < C_out; co++) {{
    unsigned int start = W_indptr[co];
    unsigned int end = W_indptr[co+1];
    unsigned idx = start;
    unsigned step = 2 * nitems;
    //std::cout << start << " : " << end << std::endl;""".
    format(args.vl, args.sp_unroll, args.sp_unroll, args.sp_unroll)]
    unroll = args.sp_unroll
    while unroll > 0:
        body = body + write_sparse_block_horizontal_split_fp16_loop(args.vl, unroll)
        unroll = unroll // 2
    body.append("""
  }
}""")

    for i in range(len(body)):
        print(body[i])



if __name__ == "__main__":
    args = parse()
    write_sparse_horizontal_split(args)
    write_sparse_vertical_split(args)
    write_sparse_block_vertical_split(args)
    write_sparse_block_horizontal_split(args)
    write_sparse_horizontal_split_fp16(args)
    write_sparse_vertical_split_fp16(args)
    write_sparse_vertical_split_fp16_fused(args)
    write_sparse_block_vertical_split_fp16(args)
    write_sparse_block_horizontal_split_fp16(args)
