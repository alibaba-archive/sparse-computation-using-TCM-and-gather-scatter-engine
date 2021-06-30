
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

#include <arm_sve.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <cstdio>
#include <cstring>
#include <iostream>

#define LOCAL_RAM_ADDR 536870912
#define GS_DATA_T float16_t
#define GS_INDEX_T unsigned short
#define GS_INDEX_SVE_T svuint16_t
#define GS_DATA_SVE_T svfloat16_t

#define GS_DATA_HACK_T float64_t

int main() {
  GS_INDEX_T indices[2048];
  GS_INDEX_T res_indices_gt[2048];
  GS_DATA_T res[2048];
  GS_DATA_T res_gt[2048]; //ground truth
  uint64_t res64[2048];
  float32_t res_32[2048];
  // GS_INDEX_T offset[8] = {1, 1, 1, 1, 1,1,1,1};
  // for (unsigned i = 0; i < rows_data; i++) {
  //   indices[i] = 4 + i;
  //   res[i] = 0;
  // }

  // allocate the local memory
  // int fd = open("/dev/mem", O_RDWR);
  // A hack to map virtual address to physical address
  // LOCAL_RAM_ADDR needs to appear twice
  // MAP_ANONYMOUS must be set
  // fd == -1
  // see mmapFunc in syscall_emul.hh
  std::cout<<"sizeof(GS_DATA_T) = "<<sizeof(GS_DATA_T)<<std::endl;
  void *addr = mmap((void *)LOCAL_RAM_ADDR, sizeof(GS_DATA_T) * 1024,
                    PROT_NONE, MAP_ANONYMOUS | MAP_PRIVATE,
                    -1, LOCAL_RAM_ADDR);

  if (addr == MAP_FAILED) {
    std::cout << "Failed to allocate on TCM" << std::endl;
  }
  std::cout << addr << std::endl;
  GS_DATA_T *data = (GS_DATA_T *)addr;

  const svbool_t true8 = svptrue_b8();
  // const svbool_t true16 = svdupq_b16(true, true, true, true,
  //                                      true, true, true, true);
  const svbool_t true16 = svptrue_b16();
  const svbool_t pfalse = svpfalse();
  const svbool_t true32 = svptrue_b32();
  const svbool_t true64 = svptrue_b64();
  const svbool_t pred16lo = svdupq_b8(false, false, false, false,
                                      false, false, false, false,
                                       true, true, true, true,
                                       true, true, true, true);
  const svbool_t pred16hi = svdupq_b8(true, true, true, true,
                                       true, true, true, true,
                                       false, false, false, false,
                                      false, false, false, false);

  size_t rows_data = 32;
  size_t rows_indices = 16;

  for (unsigned i = 0; i < rows_data; i++) {
    data[i] =2.5*i;
    // GS_DATA_T data_s = 10000.0*i;
    // memcpy(data+i,&data_s, sizeof(GS_DATA_T));
  }

  std::cout << "Data: "<<data<<std::endl;
  for (unsigned i = 0; i < rows_data; i++) {
    std::cout << data[i] << " ";
  }
  std::cout << std::endl;

  std::cout << "indices: "<<indices<<std::endl;
  for (unsigned i = 0; i < rows_indices; i++) {
    indices[i] = 2*i;//i%4+8;
    std::cout << indices[i] << " ";
  }
  std::cout << std::endl;

  size_t vlength = 8;
  size_t rows_v = rows_indices/vlength;

    // for (int i=0; i<rows_v;i++){
    //     GS_INDEX_SVE_T index = svld1_vnum(true16, indices+i*vlength, 0);
    //     svst1(true16, res_indices_gt+i*vlength, index);
    // }

    // std::cout<<"res_indices_gt: ";
    // for (unsigned i = 0; i < rows_indices; i++) {
    //     std::cout << res_indices_gt[i] << " ";
    // }

  for (int i=0; i<rows_v; i++){
    // const GS_INDEX_SVE_T index = svld1(true16, indices+i*vlength);

    std::cout<<std::endl;
    std::cout<<"i="<<i
      <<", base_1="<<indices+i*vlength
      <<", base_2="<<reinterpret_cast<int64_t>(data)
      <<", base_2_v="<<LOCAL_RAM_ADDR
      <<std::endl;
    auto inc = i*vlength;

    // GS_DATA_SVE_T A0 = svreinterpret_f16(
    //               svld1_vnum(true64, (GS_DATA_HACK_T *)(indices+inc),0)
    //               );
    GS_DATA_SVE_T A0 = spcpuFusedGather_f16(true16,indices+inc,data);


    std::cout<<"finished fused op.\n";
    svst1(true16, res+inc, A0);
    std::cout<<"finished svst1.\n";


    // const svfloat32_t A0_32_hi = svcvt_f32_z(true16, A0);
    // const svfloat32_t A0_32_lo = svcvt_f32_z(true16, A0);
    // svst1(true32, &res_32[i*vlength],A0_32_hi);
    // svst1(true32, &res_32[i*vlength+4],A0_32_lo);

  // std::cout << "Res32: ";
  // for (int j=i*vlength; j<(i+1)*vlength;j++){
  //     std::cout<<res_32[j]<<"\t";
  //   }
  // std::cout<<std::endl;
  }
  // std::cout << "Res32: ";
  // for (int j=0; j<rows_indices;j++){
  //     std::cout<<res_32[j]<<"\t";
  //   }
  // std::cout<<std::endl;

  std::cout << "Res: ";
  // GS_DATA_T res_sum = 0;
  for (unsigned i = 0; i < rows_indices; i++) {
    // std::cout<<"res: i="<<i<<"\t";
    std::cout << res[i] << " ";
  }
  std::cout << std::endl;

  // // groundtruth
  // for (int i=0; i<rows_indices; i++){
  //   res_gt[i] = data[indices[i]];
  // }

  // std::cout<<"Res Groudtruth: ";
  // for (unsigned i = 0; i < rows_indices; i++) {
  //   std::cout << res_gt[i] << " ";
  // }
  // std::cout << std::endl;

  return 0;
}
