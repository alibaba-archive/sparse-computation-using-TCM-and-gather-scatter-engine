
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
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

#include "checkpoint.h"
#include "mm.h"
#include "utils.h"

//define PRINT(x) std::cout << x << std::endl;
#define PRINT(X)

// 512MB - 512MB+32kB
#define LOCAL_RAM_ADDR 536870912  //0x20000000
#define LOCAL_RAM_ADDR_PREFETCH_MONITOR 536936448
#ifndef DIM_M
#define DIM_M 128
#endif
#ifndef DIM_N
#define DIM_N 128
#endif
#ifndef DIM_K
#define DIM_K 128
#endif
#ifndef SPARSITY
#define SPARSITY 1
#endif

#define REAL_DATA 0

#if defined(SBLOCK)
#define SPARSITY_TYPE SPARSE_BLOCK
#elif defined(SBUCKET)
#define SPARSITY_TYPE SPARSE_BUCKET
#else
#define SPARSITY_TYPE SPARSE_IRREGULAR
#endif

#if defined(SPARSE_FP32)
  #define SPARSE_DATA_T float32_t
  #define SPARSE_JOINED_INDEX_T unsigned int
#else
  #define SPARSE_DATA_T float16_t
  #define SPARSE_JOINED_INDEX_T unsigned short
#endif

#if defined(DENSE_FP16)
  #define DENSE_DATA_T float16_t
#else
  #define DENSE_DATA_T float32_t
#endif

#if defined(CONV)
  // Define convolution of size H = 14 W = 14 Cin = 256 Cout = 256
  #define CONV_H 8
  #define CONV_W 8
  #define CONV_CIN  128
  #define CONV_COUT 128
  #define CONV_KH 3
  #define CONV_KW 3

  #define A_H 1
  #define A_W (CONV_H * CONV_W * CONV_CIN)
  #define B_H CONV_COUT
  #define B_W (CONV_KH * CONV_KW * CONV_CIN)
  #define C_H 1
  #define C_W ((CONV_H - CONV_KH + 1) * (CONV_W - CONV_KW + 1) * CONV_COUT)

#else

#define B_H DIM_N
#define B_W DIM_K
#define C_H DIM_M
#define C_W DIM_N

#if defined(TRANSPOSED)
#define A_H DIM_K
#define A_W DIM_M
#else
#define A_H DIM_M
#define A_W DIM_K
#endif // TRANSPOSED

#endif // CONV

#if defined(HNUM)

#define H_NUM HNUM
#define V_NUM (vector_length/HNUM)

#else
#if defined(VNUM)
#define H_NUM (vector_length/VNUM)
#define V_NUM VNUM
#else
  #if defined(HORIZON)
    #define H_NUM vector_length
    #define V_NUM 1
  #else
    #define V_NUM vector_length  //default is vertical
    #define H_NUM 1
#endif
#endif // VNUM
#endif // HNUM

#define TCM 1
#if TCM
#define TCM_NUM_ELEM 32768
  #if defined(DENSE_FP16)
    #define TCM_NUM_BYTE_PER_ELEM 2
  #else
    #define TCM_NUM_BYTE_PER_ELEM 4
  #endif
// #define TCM_NUM_BYTE_PER_ELEM 32
#endif
int main(int argc, char **argv) {
  // allocate the local memory
  // int fd = open("/dev/mem", O_RDWR);
  // A hack to map virtual address to physical address
  // LOCAL_RAM_ADDR needs to appear twice
  // MAP_ANONYMOUS must be set
  // fd == -1
  // see mmapFunc in syscall_emul.hh
  // reset_stats();
//   std::string prefix = "/mnt/ssd/f.sun/weights/ResNet50/";
  std::string prefix = "/root/shared/zihao/compressed_format/";
//   #if defined(SBUCKET)
//     #if defined(HORIZON)
//     std::string pattern = "gs_90_8_8";
//     #else
//     std::string pattern = "gs_90_8_1";
//     #endif
//   #elif defined(SBLOCK)
//     #if defined(HORIZON)
//     std::string pattern = "block_90_1_8";
//     #else
//     std::string pattern = "block_90_8_1";
//     #endif
//   #endif  
#if defined(CONV)
    #if defined(SBUCKET)
        #if defined(HORIZON)
        std::string pattern = "resnet_gs_0.9_8_8";
        #else
        std::string pattern = "resnet_gs_0.9_8_1";
        #endif
    #elif defined(SBLOCK)
        #if defined(HORIZON)
        std::string pattern = "resnet_blk_0.9_1_8";
        #else
        std::string pattern = "resnet_blk_0.9_8_1";
        #endif
    #endif
  #else
    #if defined(SBUCKET)
        #if defined(HORIZON)
        std::string pattern = "gs_90_8_8";
        #else
        std::string pattern = "gs_90_8_1";
        #endif
    #elif defined(SBLOCK)
        #if defined(HORIZON)
        std::string pattern = "block_90_1_8";
        #else
        std::string pattern = "block_90_8_1";
        #endif
    #endif
  #endif

  std::string layer = "decoder.att_rnn.attn.linear_k.weight";
  //"layer2.0.conv2.weight";
  //layer1.0.conv1
  // layer2.0.conv2
  // decoder.att_rnn.attn.linear_k

  std::string ptr_file =
    prefix + pattern + "/PTR_" + layer + ".txt";
  std::string ptr_cmf_file =
    prefix + pattern + "/PTR_" + layer + "_cmf.txt";
  std::string v_file =
    prefix + pattern + "/V_" + layer + ".txt";

  /*
  std::string ptr_file =
    prefix + "gs_90_8_8/PTR_decoder.att_rnn.attn.linear_k.weight.txt";
  std::string ptr_cmf_file =
    prefix + "gs_90_8_8/PTR_decoder.att_rnn.attn.linear_k.weight_cmf.txt";
  std::string v_file =
    prefix + "gs_90_8_8/V_decoder.att_rnn.attn.linear_k.weight.txt";
  */
  /*
    std::string ptr_file =
      prefix + "block_90_1_8/PTR_decoder.att_rnn.attn.linear_k.weight.txt";
    std::string ptr_cmf_file =
      prefix + "block_90_1_8/PTR_decoder.att_rnn.attn.linear_k.weight_cmf.txt";
    std::string v_file =
      prefix + "block_90_1_8/V_decoder.att_rnn.attn.linear_k.weight.txt";
  */
  /*
    std::string ptr_file =
      prefix + "block_90_8_1/PTR_decoder.att_rnn.attn.linear_k.weight.txt";
    std::string ptr_cmf_file =
      prefix + "block_90_8_1/PTR_decoder.att_rnn.attn.linear_k.weight_cmf.txt";
    std::string v_file =
      prefix + "block_90_8_1/V_decoder.att_rnn.attn.linear_k.weight.txt";
  */
  #if TCM
  size_t tcm_length_byte = TCM_NUM_BYTE_PER_ELEM * TCM_NUM_ELEM;

  void *addr = mmap((void *)LOCAL_RAM_ADDR, tcm_length_byte,
                    PROT_NONE, MAP_ANONYMOUS | MAP_PRIVATE,
                    -1, LOCAL_RAM_ADDR);
  void *addr_prefetcher_monitor = mmap((void *)LOCAL_RAM_ADDR_PREFETCH_MONITOR,
                    sizeof(uint64_t),
                    PROT_NONE, MAP_ANONYMOUS | MAP_PRIVATE,
                    -1, LOCAL_RAM_ADDR_PREFETCH_MONITOR);
  if (addr == MAP_FAILED) {
    std::cout << "Failed to allocate on TCM" << std::endl;
  }
  std::cout << "Allocate the local memory to real machine memory address : "<< addr << std::endl;
  std::cout << "Allocated memory size (byte): "<< tcm_length_byte << std::endl;
  #endif
  unsigned int vector_length;
  #if defined(SPARSE_FP32)
    vector_length = svcntw();
  #else
    vector_length = svcnth();
  #endif
  if (SPARSITY_TYPE == SPARSE_IRREGULAR) {
    vector_length = 1;
  }
  std::cout<<"### HL: vector length = " <<vector_length<<std::endl;
  // Sparse matrix is transposed
#if defined(JOINED)
  #if (REAL_DATA == 1)
  std::cout<<"Initialize sparse matrix from real data..\n";
  SparseJoinedMatrix<SPARSE_DATA_T, SPARSE_JOINED_INDEX_T> *bm =
      new SparseJoinedMatrix<SPARSE_DATA_T, SPARSE_JOINED_INDEX_T>(
      B_H, B_W, SPARSITY_TYPE, H_NUM, V_NUM, false,
      ptr_file, ptr_cmf_file, v_file);
  #else
   std::cout<<"Initialize sparse matrix manually..\n";
    SparseJoinedMatrix<SPARSE_DATA_T, SPARSE_JOINED_INDEX_T> *bm =
      new SparseJoinedMatrix<SPARSE_DATA_T, SPARSE_JOINED_INDEX_T>(
      B_H, B_W, SPARSITY_TYPE, H_NUM, V_NUM, SPARSITY, false);
  #endif
  SPARSE_JOINED_INDEX_T *b_data = bm->get_joined();
  unsigned int *b_indptr = bm->get_indptr();
  std::cout << "B matrix starting address : " << std::hex
      << (unsigned long long) b_data  << std::dec << std::endl;
  std::cout << "B matrix size : " << bm->get_joined_size() << std::endl;
  #if defined(SCATTER)
    unsigned short *b_rowind = bm->get_rowind();//Prefetcher start
  #endif
  ((uint64_t*)addr_prefetcher_monitor)[0] =
      (uint64_t)b_data;
  ((uint64_t*)addr_prefetcher_monitor)[1] =
      (uint64_t)(bm->get_joined_size()*2); //store block size for prefetching
    std::cout<<"Set prefetch start addr= "<<std::hex
        << (unsigned long long)b_data<< std::dec
        <<", block size= "<<(uint64_t)(bm->get_joined_size()*2)
        <<std::endl;
#else
    #if (REAL_DATA == 1)
      std::cout<<"Initialize sparse matrix from real data..\n";
      SparseMatrix<SPARSE_DATA_T, SPARSE_JOINED_INDEX_T> *bm =
          new SparseMatrix<SPARSE_DATA_T, SPARSE_JOINED_INDEX_T>(
          B_H, B_W, SPARSITY_TYPE, H_NUM, V_NUM, SPARSITY, false,
          ptr_file, ptr_cmf_file, v_file);
    #else
      std::cout<<"Initialize sparse matrix manually..\n";
      SparseMatrix<SPARSE_DATA_T, SPARSE_JOINED_INDEX_T> *bm =
          new SparseMatrix<SPARSE_DATA_T, SPARSE_JOINED_INDEX_T>(
          B_H, B_W, SPARSITY_TYPE, H_NUM, V_NUM, SPARSITY, false);
    #endif
    SPARSE_DATA_T *b_value = bm->get_values();
    SPARSE_JOINED_INDEX_T *b_indices = bm->get_indices();
    unsigned int *b_indptr = bm->get_indptr();
    std::cout << "B matrix starting address : " << std::hex
        << (unsigned long long) b_value  << std::dec << std::endl;
    std::cout << "B matrix size : " << bm->get_size() << std::endl;
    //Prefetcher start
    // memcpy((uint64_t*)addr_prefetcher_monitor, b_value, sizeof(uint64_t));
    ((uint64_t*)addr_prefetcher_monitor)[0] =
        (uint64_t)b_value; //store start addr for prefetching
    ((uint64_t*)addr_prefetcher_monitor)[1] =
        (uint64_t)(bm->get_size()*2); //store block size for prefetching
    std::cout<<"Sparse_harness: Set prefetch start addr= "<<std::hex
        << (unsigned long long)b_value<< std::dec
        <<", block size= "<<(uint64_t)(bm->get_size()*2)
        <<std::endl;

#endif
//  exit(0);
// dump_stats
// reset_stats();
// exit(0);

//activation is a vector
#if (DIM_M == 1) || defined(CONV)
  #if TCM
    DenseMatrix<DENSE_DATA_T> *am = new DenseMatrix<DENSE_DATA_T>(addr,
                                                            A_H, A_W, false);
  #else
    DenseMatrix<DENSE_DATA_T> *am = new DenseMatrix<DENSE_DATA_T>(A_H,
      A_W, false);
  #endif
  DENSE_DATA_T *a = am->get_values();
  #if defined(SCATTER)
    //For scatter-version spMV,
    //allocate the result dense vector to the 2nd half of TCM
    DenseMatrix<DENSE_DATA_T> *cm = new DenseMatrix<DENSE_DATA_T>(
      addr + tcm_length_byte/2,
      C_H, C_W, false);
  #else
    DenseMatrix<DENSE_DATA_T> *cm = new DenseMatrix<DENSE_DATA_T>(C_H,
      C_W, false);
  #endif
  DENSE_DATA_T *c = cm->get_values();
#else
    DenseMatrix<DENSE_DATA_T> *am = new DenseMatrix<DENSE_DATA_T>(A_H,
      A_W, false);
    DENSE_DATA_T *a = am->get_values();
    DenseMatrix<DENSE_DATA_T> *cm =
        new DenseMatrix<DENSE_DATA_T>(C_H, C_W, false);
    DENSE_DATA_T *c = cm->get_values();
#endif

/********* perfetch trigger **********/
uint64_t addr_prefetch = ((uint64_t*)addr_prefetcher_monitor)[0];
// std::cout<<"Sparse_harness: Trigger prefetch, start addr to: "<<std::hex
//       << (unsigned long long)addr_prefetch<< std::dec<<std::endl;
  // dump_stats();


reset_stats();


#ifdef SCALAR
  PRINT("Scalar")
  spmm_scalar(c, a, b_value, b_indices, b_indptr, DIM_M, DIM_K, DIM_N);
#elif defined(SBLOCK)
  PRINT("Block")
  #if DIM_M == 1
    #if defined(DENSE_FP16)
        PRINT("FP16")
      # if defined(CONV)
        # if defined(HORIZON)
            #if defined(COMPUTE_FP32)
         sp_conv_block_horizontal_fp16_fp32_compute (
                          c, a, b_value, b_indices, b_indptr, CONV_H, CONV_W,
                          CONV_KH, CONV_KW, CONV_CIN, CONV_COUT,
                          vector_length);
            #else
         sp_conv_block_horizontal_fp16 (
                          c, a, b_value, b_indices, b_indptr, CONV_H, CONV_W,
                          CONV_KH, CONV_KW, CONV_CIN, CONV_COUT,
                          vector_length);

            #endif
        #else
            #if defined(COMPUTE_FP32)
        sp_conv_block_vertical_fp16_fp32_compute (
                          c, a, b_value, b_indices, b_indptr, CONV_H, CONV_W,
                          CONV_KH, CONV_KW, CONV_CIN, CONV_COUT,
                          vector_length);
            #else
        sp_conv_block_vertical_fp16 (
                          c, a, b_value, b_indices, b_indptr, CONV_H, CONV_W,
                          CONV_KH, CONV_KW, CONV_CIN, CONV_COUT,
                          vector_length);

            #endif
        #endif
      #else
       #if defined(HORIZON)
         spmv_block_horizonal_fp16(c, a, b_value, b_indices, b_indptr, DIM_K,
                               DIM_N, vector_length);
       #else
         spmv_block_vertical_fp16(c, a, b_value, b_indices, b_indptr, DIM_K,
                               DIM_N, vector_length);
       #endif
      #endif
    #else
    #if defined(SPLIT)
    PRINT("Split")
    spmv_block_vertical_split(c, a, b_value, b_indices, b_indptr, DIM_K, DIM_N,
                        vector_length);

    #else
      #if defined(FP32)
        PRINT("FP32 block version")
        spmv_block_vertical_fp32(c, a, b_value,
                                 b_indices, b_indptr, DIM_K, DIM_N,
                                 vector_length);
      #else
        spmv_block_vertical(c, a, b_value, b_indices, b_indptr, DIM_K, DIM_N,
                          vector_length);
      #endif
    #endif
    #endif
  #else
    #ifdef SPLIT
    PRINT("Split")
      #ifdef TRANSPOSED
      PRINT("Transposed")
  spmm_block_vertical_split_transposed(c, a, b_value, b_indices, b_indptr,
                    DIM_M, DIM_K, DIM_N, vector_length);
      #else
  spmm_block_vertical_split(c, a, b_value, b_indices, b_indptr,
                    DIM_M, DIM_K, DIM_N, vector_length);
      #endif
    #else
  spmm_block_vertical(c, a, b_value, b_indices, b_indptr,
                      DIM_M, DIM_K, DIM_N, vector_length);
    #endif
  #endif
#elif defined(SBUCKET)
  #if defined(CONV)
    #if defined(DENSE_FP16)
      #if defined(HORIZON)
        #if defined(COMPUTE_FP32)
      sp_conv_horizontal_fp16_fp32_compute(
                          c, a, b_data, b_indptr, CONV_H, CONV_W,
                          CONV_KH, CONV_KW, CONV_CIN, CONV_COUT,
                          vector_length);
        #else
      sp_conv_horizontal_fp16(c, a, b_data, b_indptr, CONV_H, CONV_W,
                            CONV_KH, CONV_KW, CONV_CIN, CONV_COUT,
                            vector_length);
        #endif
      #else
        #if defined(COMPUTE_FP32)
      sp_conv_vertical_fp16_fp32_compute(
                          c, a, b_data, b_indptr, CONV_H, CONV_W,
                          CONV_KH, CONV_KW, CONV_CIN, CONV_COUT,
                          vector_length);
        #else
          #if defined(FUSED)
      sp_conv_vertical_fp16_fused(c, a, b_data, b_indptr, CONV_H, CONV_W,
                            CONV_KH, CONV_KW, CONV_CIN, CONV_COUT,
                           vector_length);
          #else
      sp_conv_vertical_fp16(c, a, b_data, b_indptr, CONV_H, CONV_W,
                            CONV_KH, CONV_KW, CONV_CIN, CONV_COUT,
                           vector_length);
          #endif
        #endif

      #endif
    #else

    #endif
  #else
  assert(DIM_M == 1);
  PRINT("Bucket")
  #if defined(JOINED)
  PRINT("Joined")
    #if defined(SPLIT)
  //spmv_joined_bucket_vertical_split(c, a, b_data, b_indptr,
    //                  DIM_K, DIM_N, vector_length);

    spmv_joined_bucket_vertical_split_fp16(c, a, b_data, b_indptr,
                     DIM_K, DIM_N, vector_length);

    #else
      #if defined(SPARSE_FP32)
        spmv_joined_bucket_vertical_sparsefp32(c, a, b_data, b_indptr,
                      DIM_K, DIM_N, vector_length);
      #elif defined(DENSE_FP16)
          # if defined(HORIZON)
         //std::cout <<" run  spmv_joined_bucket_horizontal_fp16" << std::endl;
        spmv_joined_bucket_horizontal_fp16(c, a, b_data, b_indptr,
                      DIM_K, DIM_N, vector_length);

          #else
            #if defined(SCATTER)
              spmv_joined_bucket_vertical_scatter_fp16(c, a,
                            b_data, b_indptr, b_rowind,
                            DIM_K, DIM_N, vector_length);
            #else
              #if defined(FUSED)
                spmv_joined_bucket_vertical_densefp16_fused(
                  c, a, b_data, b_indptr,
                DIM_K, DIM_N, vector_length);
              #else
                spmv_joined_bucket_vertical_densefp16(c, a, b_data, b_indptr,
                              DIM_K, DIM_N, vector_length);
              #endif
            #endif
          #endif
      #else
        spmv_joined_bucket_vertical(c, a, b_data, b_indptr,
                      DIM_K, DIM_N, vector_length);
      #endif
    #endif
  #else
    #if defined(SPLIT)
  PRINT("Split")
  spmv_bucket_vertical_split2(c, a, b_value, b_indices, b_indptr,
                      DIM_K, DIM_N, vector_length);
    #else
    # if defined(DENSE_FP16)
  spmv_bucket_vertical_fp16(c, a, b_value, b_indices, b_indptr,
                      DIM_K, DIM_N, vector_length);
    #else
  spmv_bucket_vertical(c, a, b_value, b_indices, b_indptr,
                      DIM_K, DIM_N, vector_length);
    #endif
    #endif
  #endif
  #endif
#else
  assert(false);
#endif
  dump_stats();

  std::cout<<"prefetched addr = "<<addr_prefetch<<std::endl;
#if false && (DIM_M == 1)
  std::cout<<"c values:\n";
  for(size_t i=0; i<DIM_N; i++){
    std::cout<<"c["<<i<<"]="<<c[i]<<"\t";
  }
  std::cout<<"\n";
#endif
  // cal_sum_vec<float32_t>(c, DIM_N);
  //delete am;
  //delete bm;
  //delete cm;
}
