
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
#include <arm_sve.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <iostream>

#define LOCAL_RAM_ADDR 536870912
#define GS_DATA_T float
#define GS_INDEX_T unsigned
#define GS_INDEX_SVE_T svuint32_t
#define GS_DATA_SVE_T svfloat32_t
int main() {
  GS_INDEX_T indices[2048];
  GS_DATA_T res[2048];
  GS_INDEX_T offset[4] = {1, 1, 1, 1};
  for (unsigned i = 0; i < 10; i++) {
    indices[i] = 4 + i;
    res[i] = 0;
  }

  // allocate the local memory
  // int fd = open("/dev/mem", O_RDWR);
  // A hack to map virtual address to physical address
  // LOCAL_RAM_ADDR needs to appear twice
  // MAP_ANONYMOUS must be set
  // fd == -1
  // see mmapFunc in syscall_emul.hh
  void *addr = mmap((void *)LOCAL_RAM_ADDR, 32 * 1024,
                    PROT_NONE, MAP_ANONYMOUS | MAP_PRIVATE,
                    -1, LOCAL_RAM_ADDR);

  if (addr == MAP_FAILED) {
    std::cout << "Failed to allocate on TCM" << std::endl;
  }
  std::cout << addr << std::endl;
  GS_DATA_T *data = (GS_DATA_T *)addr;

  const svbool_t pred32 = svptrue_b32();

  for (unsigned i = 0; i < 8; i++) {
    data[i] =1.5 + i;
  }

  std::cout << "Data: ";
  for (unsigned i = 0; i < 8; i++) {
    std::cout << data[i] << " ";
  }
  std::cout << std::endl;

  std::cout << "Index: ";
  for (unsigned i = 0; i < 4; i++) {
    indices[i] = 2*i;
    std::cout << indices[i] << " ";
  }
  std::cout << std::endl;
  const GS_INDEX_SVE_T index = svld1(pred32, indices);

  GS_DATA_SVE_T A0 = svld1_gather_index(pred32, data, index);

  svst1(pred32, res, A0);
  std::cout << "Res: ";
  for (unsigned i = 0; i < 4; i++) {
    std::cout << res[i] << " ";
  }
  std::cout << std::endl;

  return res[0];
}
