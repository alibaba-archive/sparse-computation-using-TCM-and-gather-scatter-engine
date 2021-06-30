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
    parser.add_argument('--co_i', type=int, required=True)
    parser.add_argument('--ci_i', type=int, required=True)
    parser.add_argument('--kh_i', type=int, required=True)
    parser.add_argument('--kw_i', type=int, required=True)
    parser.add_argument('--vl', type=int, required=True, help="vector length")
    parser.add_argument('--fp16', action="store_true")
    parser.add_argument('--fp32', action="store_true")
    args = parser.parse_args()
    return args

# def write_dense_conv_split_fp16(args):
#     body = ["""
#     #include <cassert>
#     #include <iostream>
#     #include "mm.h"

# void dense_conv_fp16(float16_t *C, float16_t const *A_padded,
#                   float16_t const *B,
#                   int A_h, int A_w, int W_h, int W_w, unsigned int C_in,
#                   unsigned int C_out,
#                   int nitems) {{

#   assert (svcnth() == 8);
#   assert (W_h == {});
#   assert (W_w == {});
#   assert (C_in % {} == 0);
#   assert ({} == 1);
#   std::cout <<" start to run fp16 " << std::endl;
#   const svbool_t pred16 = svptrue_b16();
#   // svfloat16_t acc = svdup_f16(0);
#   unsigned int H_out = A_h - W_h + 1;
#   unsigned int W_out = A_w - W_w + 1;""".
#   format(args.kh, args.kw, args.ci_i, args.co_i)]
#     body.append("""
#   unsigned offset;
#   const float16_t *B_addr, *addr;
#   for(unsigned int co = 0; co < C_out; co+= {}) {{
#     for (unsigned int ci = 0; ci < C_in; ci+= {}) {{
# """.format(args.co_i, args.ci_i))
#     for kh in range(0, args.kh):
#         for kw in range(0, args.kw):
#             body.append("""
#       offset = co*C_in*W_h*W_w+{}*W_w*C_in+{}*C_in+ci;
#       B_addr = &B[offset];""".
#                 format(kh, kw))
#             for i in range (0, args.ci_i // args.vl):
#                 body.append("""
#       svfloat16_t B_{}_{}_{} = svld1(pred16, &B_addr[{}]);""".
#                     format(kh, kw, i, i * args.vl))
#     body.append("""
#       for (unsigned int h = 0; h < H_out; h++){
#         for(unsigned w = 0; w < W_out; w++){
#           unsigned O_addr = (h*W_out+w)*C_out+co;
#           float16_t const * A_addr = &A_padded[(h*A_w+w)*C_in+ci];
# """)
#     body.append("""          svfloat16_t acc = svdup_f16(0);""")
#     for kh in range(0, args.kh):
#         for kw in range(0, args.kw):
#             for i in range (0, args.ci_i // args.vl):
#                 body.append(
# """           addr = &A_addr[({}*A_w+{})*C_in+{}];
#           svfloat16_t A_{}_{}_{} = svld1(pred16,addr);""".
#            format(kh, kw,i, kh, kw, i * args.vl))
#                 body.append(
# """           acc = svmla_x(pred16, acc, A_{}_{}_{}, B_{}_{}_{});""".
#             format(kh, kw, i, kh, kw, i))
#     body.append("""
#           float16_t r = svaddv(pred16, acc);
#           float partial = C[O_addr];
#           float r32 = (float)r;
#           float p32 = (float)partial;
#           // hack we do not add the bias as fadd does not exit on half
#           float res32 =  r;
#           float16_t res16 = (float16_t) res32;
#           C[O_addr] = res32;
#         }
#       }
#     }
#   }
# }
# """)
#     for i in range(len(body)):
#         print(body[i])

def write_dense_conv_split_fp16(args):
    body = ["""
    #include <cassert>
    #include <iostream>
    #include "mm.h"

void dense_conv_fp16(float16_t *C, float16_t const *A_padded,
                  float16_t const *B,
                  int A_h, int A_w, unsigned W_h, unsigned W_w, unsigned C_in,
                  unsigned int C_out,
                  int nitems) {{
  assert (svcnth() == 8);
  assert (W_h % {} == 0);
  assert (W_w % {} == 0);
  assert (C_in % {} == 0);
  assert ({} == 1);
  const svbool_t pred16 = svptrue_b16();
  const svbool_t pred32 = svptrue_b32();
  // const svbool_t pred16lo = svdupq_b16(false, false, false, false,
  //                                       true, true, true, true);
  // const svbool_t pred16hi = svdupq_b16(true, true, true, true,
  //                                       false, false, false, false);
  // svfloat16_t acc = svdup_f16(0);
  unsigned int H_out = A_h - W_h + 1;
  unsigned int W_out = A_w - W_w + 1;
  std::cout << ":" << std::endl;""".
  format(args.kh_i, args.kw_i, args.ci_i, args.co_i)]
    body.append("""
  unsigned offset;
  const float16_t *B_addr, *addr;
  for(unsigned int co = 0; co < C_out; co+= {}) {{
    for (unsigned int ci = 0; ci < C_in; ci+= {}) {{
      for (unsigned kh = 0; kh < W_h; kh+= {}) {{
        for (unsigned kw = 0; kw < W_w; kw+= {}) {{
            // std::cout << co << ":" << std::endl;
            // std::cout << co << ":" 
            //           << ci << ":"  
            //           << kh << ":"  
            //           << kw << ":" << std::endl;
""".format(args.co_i, args.ci_i, args.kh_i, args.kw_i))
    for kh in range(0, args.kh_i):
        for kw in range(0, args.kw_i):
            body.append("""
      offset = co*C_in*W_h*W_w+{}*W_w*C_in+{}*C_in+ci;
      B_addr = &B[offset];""".
                format(kh, kw))
            for i in range (0, args.ci_i // args.vl):
                body.append("""
      svfloat16_t B_{}_{}_{} = svld1(pred16, &B_addr[{}]);""".
            format(kh, kw, i, i * args.vl))
    body.append("""
      for (unsigned int h = 0; h < H_out; h++){
        for(unsigned w = 0; w < W_out; w++){
          unsigned O_addr = (h*W_out+w)*C_out+co;
          float16_t const * A_addr = &A_padded[(h*A_w+w)*C_in+ci];
""")
    body.append("""          svfloat16_t acc = svdup_f16(0);""")
    for kh in range(0, args.kh_i):
        for kw in range(0, args.kw_i):
            for i in range (0, args.ci_i // args.vl):
                body.append(
"""           addr = &A_addr[({}*A_w+{})*C_in+{}];
          svfloat16_t A_{}_{}_{} = svld1(pred16,addr);""".
           format(kh, kw,i, kh, kw, i))
                body.append(
"""        acc = svmla_x(pred16, acc, A_{}_{}_{}, B_{}_{}_{}); //perform mla""".
           format(kh, kw, i, kh, kw, i))
    body.append("""
          float16_t r = svaddv(pred16, acc);
          float16_t res16 = (float16_t) r;
          C[O_addr] = res16;
        }
      }
    }
    }
    }
  }
}
""")
    for i in range(len(body)):
        print(body[i])


def write_dense_conv_split_fp32(args):
    body = ["""
    #include <cassert>
    #include <iostream>
    #include "mm.h"

void dense_conv_fp16(float16_t *C, float16_t const *A_padded,
                  float16_t const *B,
                  int A_h, int A_w, unsigned W_h, unsigned W_w, unsigned C_in,
                  unsigned int C_out,
                  int nitems) {{
  assert (svcnth() == 8);
  assert (W_h % {} == 0);
  assert (W_w % {} == 0);
  assert (C_in % {} == 0);
  assert ({} == 1);
  const svbool_t pred16 = svptrue_b16();
  const svbool_t pred32 = svptrue_b32();
  // const svbool_t pred16lo = svdupq_b16(false, false, false, false,
  //                                       true, true, true, true);
  // const svbool_t pred16hi = svdupq_b16(true, true, true, true,
  //                                       false, false, false, false);
  // svfloat16_t acc = svdup_f16(0);
  unsigned int H_out = A_h - W_h + 1;
  unsigned int W_out = A_w - W_w + 1;
  std::cout << ":" << std::endl;""".
  format(args.kh_i, args.kw_i, args.ci_i, args.co_i)]
    body.append("""
  unsigned offset;
  const float16_t *B_addr, *addr;
  for(unsigned int co = 0; co < C_out; co+= {}) {{
    for (unsigned int ci = 0; ci < C_in; ci+= {}) {{
      for (unsigned kh = 0; kh < W_h; kh+= {}) {{
        for (unsigned kw = 0; kw < W_w; kw+= {}) {{
""".format(args.co_i, args.ci_i, args.kh_i, args.kw_i))
    for kh in range(0, args.kh_i):
        for kw in range(0, args.kw_i):
            body.append("""
      offset = co*C_in*W_h*W_w+{}*W_w*C_in+{}*C_in+ci;
      B_addr = &B[offset];""".
                format(kh, kw))
            for i in range (0, args.ci_i // args.vl):
                body.append("""
      svfloat16_t B_{}_{}_{} = svld1(pred16, &B_addr[{}]);
      svfloat16_t B_{}_{}_{}_odd16 = svtrn2(B_{}_{}_{}, B_{}_{}_{});
      const svfloat32_t B_{}_{}_{}_even = svcvt_f32_x(pred16, B_{}_{}_{});
      const svfloat32_t B_{}_{}_{}_odd = svcvt_f32_x(pred16, B_{}_{}_{}_odd16);""".
            format(kh, kw, i, i * args.vl,
                   kh, kw, i, kh, kw, i, kh, kw, i, kh, kw, i, kh, kw, i, kh, kw, i, kh, kw, i))
    body.append("""
      for (unsigned int h = 0; h < H_out; h++){
        for(unsigned w = 0; w < W_out; w++){
          unsigned O_addr = (h*W_out+w)*C_out+co;
          float16_t const * A_addr = &A_padded[(h*A_w+w)*C_in+ci];
""")
    body.append("""          svfloat32_t acc = svdup_f32(0);
                             svfloat32_t acc1 = svdup_f32(0);""")
    for kh in range(0, args.kh_i):
        for kw in range(0, args.kw_i):
            for i in range (0, args.ci_i // args.vl):
                body.append(
"""           addr = &A_addr[({}*A_w+{})*C_in+{}];
          svfloat16_t A_{}_{}_{} = svld1(pred16,addr);
          svfloat16_t A_{}_{}_{}_odd16 = svtrn2(A_{}_{}_{}, A_{}_{}_{});
          const svfloat32_t A_{}_{}_{}_even = svcvt_f32_x(pred16, A_{}_{}_{});
          const svfloat32_t A_{}_{}_{}_odd = svcvt_f32_x(pred16, A_{}_{}_{}_odd16);""".
           format(kh, kw,i, kh, kw, i, kh, kw, i,
                kh, kw, i, kh, kw, i, kh, kw, i ,
                kh, kw, i, kh, kw, i, kh, kw, i ))
                body.append(
"""        acc = svmla_x(pred32, acc, A_{}_{}_{}_even, B_{}_{}_{}_even); //perform mla
           acc1 = svmla_x(pred32, acc1, A_{}_{}_{}_odd, B_{}_{}_{}_odd); //perform mla""".
           format(kh, kw, i, kh, kw, i,
                  kh, kw, i, kh, kw, i,))
    body.append("""
          acc = svadd_z(pred32, acc, acc1);
          float32_t r = svaddv(pred32, acc);
          float16_t res16 = (float16_t) r;
          C[O_addr] = res16;
        }
      }
    }
    }
    }
  }
}
""")
    for i in range(len(body)):
        print(body[i])



if __name__ == "__main__":
    args = parse()
    if args.fp16:
            write_dense_conv_split_fp16(args)
    elif args.fp32:
            write_dense_conv_split_fp32(args)
    else:
        pass
