


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
#include <unistd.h>

#include <cassert>
#include <ctime>
#include <fstream>
#include <iostream>
#include <memory>
#include <utility>

#ifndef __MATRIX_UTILS__
#define __MATRIX_UTILS__

void flush_cache(unsigned int size, int step);

template<typename TYPE>
class Blob {
public:
  Blob(unsigned int size);
  Blob(unsigned int size, unsigned int alignment);
  Blob(void *start_addr, unsigned int size);

  ~Blob() {
    if (!_is_absolute_addr) {
      free(_value);
    }
  }

  inline TYPE *get() { return _p; }
  inline unsigned int get_alignment() { return _alignment; }
  void randomize();
  void regular_init();
  void init_from_array(const TYPE* array, unsigned int array_size);
  void init_from_array(const float* array, unsigned int array_size);
  void init_joined_from_array(const float* values,
                              const unsigned short* indices,
                              unsigned int v_size);
  inline TYPE get(int i) { return _p[i]; }
  inline unsigned long long get_size() {  return _size; }

private:
  void *_value;
  TYPE *_p;
  unsigned int _alignment;
  unsigned int _size;
  // do not allocate and free
  bool _is_absolute_addr;

  void init(unsigned int size, unsigned int alignment);
};


template<typename TYPE>
class DenseMatrix {
public:
  DenseMatrix(unsigned int rows, unsigned int cols, bool random);
  DenseMatrix(void *start_addr,
              unsigned int rows,
              unsigned int cols,
              bool random);
  ~DenseMatrix();

  inline unsigned int get_rows() { return _rows; }
  inline unsigned int get_cols() { return _cols; }
  inline TYPE *get_values() { return _values->get(); }
  inline unsigned int get_size() { return _values->get_size(); }

private:
  unsigned int _rows;
  unsigned int _cols;
  Blob<TYPE> *_values;
};

enum SparseType {
  SPARSE_IRREGULAR,
  SPARSE_BLOCK,
  SPARSE_BUCKET,
};

template<typename TYPE, typename TYPE_INDEX>
class SparseMatrix {
public:
  SparseMatrix(unsigned int m, unsigned int n, SparseType type,
               unsigned short hitems, unsigned short vitems,
               float sparsity, bool random);
  SparseMatrix(unsigned int m, unsigned int n, SparseType type,
               unsigned short hitems, unsigned short vitems,
               float sparsity, bool random,
            std::string ptr_file, std::string ptr_cmf_file, std::string v_file);
  ~SparseMatrix();

  TYPE *get_values() { return _values->get(); }
  TYPE_INDEX *get_indices() { return _indices->get(); }
  unsigned int *get_indptr() { return _indptr->get(); }
  unsigned long long get_size() { return _values->get_size(); }

private:
  Blob<TYPE> *_values;
  Blob<TYPE_INDEX> *_indices;
  Blob<unsigned int> *_indptr;
  unsigned int _rows;
  unsigned int _cols;
  unsigned int _nnz;
  SparseType _type;
  // the number of horizontal and vertical items in one block or bucket
  unsigned short _hitems;
  unsigned short _vitems;
  float _sparsity;
};

/* Index is assumed to be unsigned short */
template<typename TYPE, typename TYPE_INDEX>
class SparseJoinedMatrix {
public:
  SparseJoinedMatrix(unsigned int m, unsigned int n, SparseType type,
               unsigned short hitems, unsigned short vitems,
               float sparsity, bool random);
  SparseJoinedMatrix(unsigned int m, unsigned int n, SparseType type,
               unsigned short hitems, unsigned short vitems,
               bool random,
               std::string ptr_file, std::string ptr_cmf_file,
               std::string v_file);
  ~SparseJoinedMatrix();

  TYPE_INDEX *get_joined() { return _data->get(); }
  unsigned int *get_indptr() { return _indptr->get(); }

  unsigned long long get_joined_size() { return _data->get_size(); }

#if defined(SCATTER)
  TYPE_INDEX *get_rowind(){return _rowind->get();}
#endif

private:
  Blob<TYPE_INDEX> *_data;
  Blob<unsigned int> *_indptr;
  unsigned int _rows;
  unsigned int _cols;
  unsigned int _nnz;
  SparseType _type;
  // the number of horizontal and vertical items in one block or bucket
  unsigned short _hitems;
  unsigned short _vitems;
  float _sparsity;

#if defined(SCATTER)
  Blob<TYPE_INDEX> *_rowind;
#endif

};

template<typename TYPE>
void Blob<TYPE>::init(unsigned int size, unsigned int alignment) {
  _value = (TYPE *)calloc(size, sizeof(TYPE));
  void *value = _value;
  std::size_t s = size;
  unsigned int align = alignment * sizeof(TYPE);
  if (std::align(align, size, value, s) == nullptr) {
    free(_value);
    std::size_t new_size = size + align - 1;
    _value = (TYPE *)calloc(new_size, sizeof(TYPE));
    value = std::align(align, size, value, new_size);
    assert(value != nullptr);
  }
  _p = reinterpret_cast<TYPE*>(value);
  _alignment = alignment;
  _size = size;
  _is_absolute_addr = false;
  /*
  for (unsigned int i=0; i<_size; i++){
    _p[i] = 0;// (TYPE)1.0;
              //HLI: looks like gem5 arm-sve doesn't support fmov instructions
  }
  */
}

template<typename TYPE>
Blob<TYPE>::Blob(unsigned int size, unsigned int alignment) {
  init(size, alignment);
}

template<typename TYPE>
Blob<TYPE>::Blob(unsigned int size) {
  init(size, sizeof(TYPE));
}

template<typename TYPE>
Blob<TYPE>::Blob(void *start_addr,
                 unsigned int size) {
  _is_absolute_addr = true;
  _value = start_addr;
  _size = size;
  _p = (TYPE *)_value;
  _alignment = sizeof(TYPE);
}

template<typename TYPE>
void Blob<TYPE>::randomize() {
  // Not implemented
  assert(false);
}
template<>
void Blob<float16_t>::randomize() {
  for (unsigned int i = 0; i < _size; i++) {
    // between -2 and 2
    _p[i] = (rand() % 1000 - 500) / 250;
  }
}
template<typename TYPE>
void Blob<TYPE>::regular_init() {
  // give value when random is not supported on current gem5 isa
  // TYPE sum = 0;
  for (unsigned int i = 0; i < _size; i++) {
    _p[i] = 0.005*i;//(TYPE)i; //HL: initialize
    // printf("initialize dense v, _p[%d]=%f\n",i,_p[i]);
  }
  std::cout<<"finished regular initialization."<<std::endl;
}
template<typename TYPE>
void Blob<TYPE>::init_from_array(const TYPE* array, unsigned int array_size) {
  std::cout << "Required: " << this->get_size()
            << "  Measured: " << array_size << std::endl;
  assert(this->get_size() >= array_size);
  for (unsigned int i = 0; i < array_size; i++) {
    _p[i] = (TYPE)array[i];
  }
}

template<>
void Blob<float16_t>::init_from_array(const float* array, unsigned int array_size) {
  std::cout << "Required: " << this->get_size()
            << "  Measured: " << array_size << std::endl;
  assert(this->get_size() >= array_size);
  for (unsigned int i = 0; i < array_size; i++) {
    _p[i] = (float16_t)array[i];
  }
}

template<typename TYPE>
void Blob<TYPE>::init_joined_from_array(const float* values,
    const unsigned short* indices, unsigned int v_size) {
  std::cout << "Required: " << this->get_size()
            << "  Measured: " << v_size*2 << std::endl;
  assert(this->get_size() >= v_size * 2);
  float16_t tmp = 0.0;
  unsigned int vector_length = svcnth();
  for (unsigned int i = 0; i < v_size*2 / vector_length; i ++) {
    for (unsigned int cnt = 0; cnt < vector_length; cnt ++){
      //values
      if(i % 2 == 0){
        tmp = (float16_t)values[(i/2)*vector_length + cnt];
        _p[i*vector_length + cnt] = reinterpret_cast<unsigned short &>(tmp);
      }
      //indices
      else{
        _p[i*vector_length + cnt] = indices[(i/2)*vector_length + cnt];
      }
    }
  }
}


template<typename TYPE>
DenseMatrix<TYPE>::DenseMatrix(unsigned int m, unsigned int n, bool random) {
  _values = new Blob<TYPE>(m * n);
  if (random) {
    _values->randomize();
  }
  else{
    _values->regular_init();
  }
  flush_cache(2 * 1024 * 1024, 16);
  std::cout << "Init dense matrix in sysmem of (" << m << ", " << n << ")"
    << std::endl;
}

template<typename TYPE>
DenseMatrix<TYPE>::DenseMatrix(void *start_addr,
                               unsigned int m,
                               unsigned int n,
                               bool random) {
  _values = new Blob<TYPE>(start_addr, m * n);
  if (random) {
    _values->randomize();
  }
  else{
    _values->regular_init();
  }
  std::cout << "Init dense matrix in TCM of (" << m << ", " << n << ")"
    << std::endl;
}

template<typename TYPE>
DenseMatrix<TYPE>::~DenseMatrix() {
  delete _values;
}

template<typename TYPE, typename TYPE_INDEX>
SparseMatrix<TYPE, TYPE_INDEX>::SparseMatrix(unsigned int m, unsigned int n,
                                 SparseType type,
                                 unsigned short hitems, unsigned short vitems,
                                 float sparsity, bool random) {
  assert(sparsity <= 1.0);
  assert(n % vitems == 0);
  assert(m % hitems == 0);

  _rows = m / vitems;
  _cols = n / hitems;
  _type = type;
  _hitems = hitems;
  _vitems = vitems;
  _sparsity = sparsity;

  unsigned long long size = m * n;
  unsigned long long nnz = (int)(size * sparsity);
  nnz = (int)(nnz / (hitems * vitems)) * (hitems * vitems);
  int alignment = hitems * vitems;
  unsigned int num_indices = nnz / (hitems * vitems);

  _values = new Blob<TYPE>(nnz, alignment);
  if (type == SPARSE_BLOCK) {
    _indices = new Blob<TYPE_INDEX>(num_indices);
  } else {
    _indices = new Blob<TYPE_INDEX>(nnz,
                                        alignment * sizeof(TYPE_INDEX));
  }
  _indptr = new Blob<unsigned int>(_rows + 1);

  // randomly initialize the indptr array
  unsigned int *indptr = _indptr->get();
  indptr[0] = 0;
  indptr[_rows] = num_indices;
  unsigned long long count = 0;
  for (unsigned i = 1; i < _rows; i++) {
    if (count < num_indices) {
      int num = (int)(_cols * sparsity);
      if (count + num > num_indices) {
        count = num_indices;
      } else {
        count += num;
      }
    }
    indptr[i] = count;
  }
  TYPE_INDEX *indices = _indices->get();
  for (unsigned i = 0; i < _rows; i++) {
    unsigned start = indptr[i];
    unsigned end = indptr[i+1];
    unsigned count = 0;
    for (unsigned j = start; j < end; j++) {
      indices[j] = count++ % _cols;
    }
  }
  flush_cache(nnz, 16);

  /*
  for (unsigned i = 0; i <= _rows; i++) {
    std::cout << indptr[i] << std::endl;
  }

  for (unsigned i = 0; i < nnz; i++) {
    std::cout << indices[i] << std::endl;
  }
  */

  if (random) {
    // not implemented
    assert(false);
  }
  std::cout << "Init sparse matrix of (" << m << ", " << n << ")\n";
  std::cout << "\trow: " << _rows << ", cols: " << _cols;
  std::cout << "\ttype: " << (_type == SPARSE_IRREGULAR ? "irregular" :
                             _type == SPARSE_BLOCK ? "block" :
                             _type == SPARSE_BUCKET ? "bucket" :
                             "unknown") << std::endl;
  std::cout << "\thitems: " << _hitems << ", vitems: " << _vitems << std::endl;
  std::cout << "\tsparsity: " << _sparsity << std::endl;

}


//Initialize from real parameters
template<typename TYPE, typename TYPE_INDEX>
SparseMatrix<TYPE, TYPE_INDEX>::SparseMatrix(unsigned int m, unsigned int n,
                                 SparseType type,
                                 unsigned short hitems, unsigned short vitems,
                                 float sparsity, bool random,
                                 std::string ptr_file,
                                 std::string ptr_cmf_file,
                                 std::string v_file) {
  assert(sparsity <= 1.0);
  assert(n % vitems == 0);
  assert(m % hitems == 0);

  _rows = m / vitems;
  _cols = n / hitems;
  _type = type;
  _hitems = hitems;
  _vitems = vitems;
  _sparsity = sparsity;

  unsigned long long size = m * n;
  unsigned long long nnz = (int)(size * sparsity);
  nnz = (int)(nnz / (hitems * vitems)) * (hitems * vitems);
  int alignment = hitems * vitems;
  unsigned int num_indices = nnz / (hitems * vitems);


  //Read from file
  unsigned int ptr_num = _rows + 1;

  clock_t time_start = clock();
  FILE  *fptr_cmf;
  FILE  *fv;
  fptr_cmf = fopen(ptr_cmf_file.c_str() ,"rt+");
  std::cout << ptr_cmf_file.c_str() << std::endl;
  std::cout << v_file.c_str() << std::endl;
  unsigned int *ptr_cmf = (unsigned int *)calloc(ptr_num, sizeof(unsigned int));
  float *values = (float *)calloc(nnz, sizeof(float));
  unsigned short *indices = (unsigned short *)calloc(nnz, sizeof(unsigned short));
  unsigned int shape[2] = {0, 0};
  unsigned int num1 = 0, num2 = 0;
  unsigned int ptr_cnt = 0, v_cnt = 0, idx_cnt = 0;

  fscanf(fptr_cmf, "%d ", &shape[0]);
  fscanf(fptr_cmf, "%d\n", &shape[1]);
  std::cout << "b_shape: " << shape[0]
            << ", " << shape[1] << std::endl;
  assert(shape[0] == m);
  assert(shape[1] == n);
  fscanf(fptr_cmf, "%d\n", &num1);
  fscanf(fptr_cmf, "%d\n", &num2);

  ptr_cmf[0] = 0;
  ptr_cnt++;
  // for (unsigned int i = 1; i <= ptr_num-1; i++){
  while (!feof(fptr_cmf)){
    fscanf(fptr_cmf, "%d ", &ptr_cmf[ptr_cnt]);
    ptr_cnt++;
  }
  fclose(fptr_cmf);
  std::cout << "Read ptr_cmf_file finished. size: " << ptr_cnt << ". ";
  clock_t time_end = clock();
  std::cout << "Time use: "
            << 1000*(time_end-time_start)/(double)CLOCKS_PER_SEC
            << "s" << std::endl;

  clock_t time_start2 = clock();
  unsigned int i = 0;

  // if binary exit;
  std::string idx_v_cnt_filename = v_file + ".idx_v_cnt.dat";
  std::string idx_bin_filename   = v_file + ".idx_bin.dat";
  std::string v_bin_filename     = v_file + ".v_bin.dat";
  bool exist_bin_file = false;
  // std::cout << idx_v_cnt_filename.c_str() << std::endl;
  // std::cout << idx_bin_filename.c_str() << std::endl;
  // std::cout << v_bin_filename.c_str() << std::endl;
  // std::cout << access(idx_v_cnt_filename.c_str(), 0) << std::endl;
  // std::cout << access(idx_bin_filename.c_str(), 0) << std::endl;
  // std::cout << access(v_bin_filename.c_str(), 0) << std::endl;
  if (access(idx_v_cnt_filename.c_str(), 0) == 0
      && access(idx_bin_filename.c_str(), 0) == 0
      && access(v_bin_filename.c_str(), 0) == 0 ) {
        exist_bin_file = true;
  }
  else {
    exist_bin_file = false;
  }

  unsigned int vector_length = svcnth();
  //if bin file does not exist, generate bin file.
  if (!exist_bin_file) {
    std::cout << "Bin file does not exist, generating ..." << std::endl;
    fv = fopen(v_file.c_str() ,"rt+");
    // while(i < 200){
    while(!feof(fv)){
      if (vector_length == 16){
        fscanf(fv, "%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n\
        %hu %hu %hu %hu %hu %hu %hu %hu %hu %hu %hu %hu %hu %hu %hu %hu\n",
            &values[i*16], &values[i*16+1], &values[i*16+2], &values[i*16+3],
            &values[i*16+4], &values[i*16+5], &values[i*16+6],
            &values[i*16+7],
            &values[i*16+8], &values[i*16+9], &values[i*16+10],
            &values[i*16+11],
            &values[i*16+12], &values[i*16+13], &values[i*16+14],
            &values[i*16+15],
            &indices[i*16], &indices[i*16+1], &indices[i*16+2],
            &indices[i*16+3],
            &indices[i*16+4], &indices[i*16+5], &indices[i*16+6],
            &indices[i*16+7],
            &indices[i*16+8], &indices[i*16+9], &indices[i*16+10],
            &indices[i*16+11],
            &indices[i*16+12], &indices[i*16+13], &indices[i*16+14],
            &indices[i*16+15]);

        v_cnt += 16;
        idx_cnt += 16;

      }
      else if (vector_length == 8){
        fscanf(fv, "%f %f %f %f %f %f %f %f\n\
        %hu %hu %hu %hu %hu %hu %hu %hu\n",
            &values[i*8], &values[i*8+1], &values[i*8+2], &values[i*8+3],
            &values[i*8+4], &values[i*8+5], &values[i*8+6], &values[i*8+7],
            &indices[i*8], &indices[i*8+1], &indices[i*8+2], &indices[i*8+3],
            &indices[i*8+4], &indices[i*8+5], &indices[i*8+6],
            &indices[i*8+7]);

        v_cnt += 8;
        idx_cnt += 8;
      }


      // if(i % 100 == 0){
      //   std::cout << i*2 << std::endl;
      //   time_start_reading = time_end_reading;
      //   time_end_reading = clock();
      //   std::cout << "time use: "
      //             << 1000*(time_end_reading-time_start_reading)/
      // (double)CLOCKS_PER_SEC << "s" << std::endl;
      // }
      i ++;
    }
    fclose(fv);

    // save v_cnt and idx_cnt
    std::ofstream cntoutFile(idx_v_cnt_filename);
    cntoutFile << "idx_cnt: "<< idx_cnt << std::endl;
    cntoutFile.close();

    // save array as bin file
    std::ofstream idxoutFile(idx_bin_filename,
      std::ios::out | std::ios::binary);
    for (unsigned int i = 0; i < idx_cnt; i++){
        idxoutFile.write((char*)&indices[i], sizeof(unsigned short));
    }
    idxoutFile.close();
    std::cout << "save idx_array as bin file" << std::endl;

    // save array as bin file
    std::ofstream voutFile(v_bin_filename, std::ios::out | std::ios::binary);
    for (unsigned int i = 0; i < v_cnt; i++){
        voutFile.write((char*)&values[i], sizeof(float));
    }
    voutFile.close();
    std::cout << "save v_array as bin file" << std::endl;
  }
  else {
    std::cout << "Loading from bin file!" << std::endl;
    // save v_cnt and idx_cnt
    FILE  *f_idx_v_cnt = fopen(idx_v_cnt_filename.c_str() ,"rt+");
    fscanf(f_idx_v_cnt, "%d", &idx_cnt);
    fclose(f_idx_v_cnt);
    v_cnt = 8 * idx_cnt;
    std::cout << "idx_cnt: "<< idx_cnt << std::endl;
    std::cout << "v_cnt:   "<< v_cnt << std::endl;

    // read idx_array from bin file
    std::ifstream idxinFile(idx_bin_filename, std::ios::in | std::ios::binary);
    if(!idxinFile) {
        std::cout << "bin file not exist" << std::endl;
        assert(0);
    }
    int t = 0;
    while(idxinFile.read((char*)&indices[t], sizeof(unsigned short))) {
      t++;
    }
    idxinFile.close();

    // read v_array from bin file
    std::ifstream vinFile(v_bin_filename, std::ios::in | std::ios::binary);
    if(!vinFile) {
        std::cout << "bin file not exist" << std::endl;
        assert(0);
    }
    t = 0;
    while(vinFile.read((char*)&values[t], sizeof(float))) {
      t++;
    }
    vinFile.close();
    // std::cout << "Load array from bin file" << ". ";
    // clock_t time_end_bin = clock();
    // std::cout << "Time use: "
    //           << 1000*(time_end_bin-time_start_bin)/(double)CLOCKS_PER_SEC
    // << "s" << std::endl;

    clock_t time_end2 = clock();
    std::cout << "Load finished, time use: "
              << 1000*(time_end2-time_start2)/(double)CLOCKS_PER_SEC
              << "s" << std::endl;


  }

  nnz = v_cnt;
  num_indices = idx_cnt;

  _values = new Blob<TYPE>(nnz, alignment);
  _values->init_from_array(values, nnz);
  if (type == SPARSE_BLOCK) {
    _indices = new Blob<TYPE_INDEX>(num_indices);
    _indices->init_from_array(indices, num_indices);
  } else {
    _indices = new Blob<TYPE_INDEX>(nnz,
                                        alignment * sizeof(TYPE_INDEX));
    _indices->init_from_array(indices, num_indices);
  }
  _indptr = new Blob<unsigned int>(_rows + 1);
  _indptr->init_from_array(ptr_cmf, _rows + 1);

  // // randomly initialize the indptr array
  // unsigned int *indptr = _indptr->get();
  // indptr[0] = 0;
  // indptr[_rows] = num_indices;
  // unsigned long long count = 0;
  // for (unsigned i = 1; i < _rows; i++) {
  //   if (count < num_indices) {
  //     int num = (int)(_cols * sparsity);
  //     if (count + num > num_indices) {
  //       count = num_indices;
  //     } else {
  //       count += num;
  //     }
  //   }
  //   indptr[i] = count;
  // }
  // TYPE_INDEX *indices = _indices->get();
  // for (unsigned i = 0; i < _rows; i++) {
  //   unsigned start = indptr[i];
  //   unsigned end = indptr[i+1];
  //   unsigned count = 0;
  //   for (unsigned j = start; j < end; j++) {
  //     indices[j] = count++ % _cols;
  //   }
  // }
  flush_cache(nnz, 16);

  /*
  for (unsigned i = 0; i <= _rows; i++) {
    std::cout << indptr[i] << std::endl;
  }

  for (unsigned i = 0; i < nnz; i++) {
    std::cout << indices[i] << std::endl;
  }
  */

  if (random) {
    // not implemented
    assert(false);
  }
  std::cout << "Init sparse matrix of (" << m << ", " << n << ")\n";
  std::cout << "\trow: " << _rows << ", cols: " << _cols;
  std::cout << "\ttype: " << (_type == SPARSE_IRREGULAR ? "irregular" :
                             _type == SPARSE_BLOCK ? "block" :
                             _type == SPARSE_BUCKET ? "bucket" :
                             "unknown") << std::endl;
  std::cout << "\thitems: " << _hitems << ", vitems: " << _vitems << std::endl;
  std::cout << "\tsparsity: " << _sparsity << std::endl;

}


template<typename TYPE,typename TYPE_INDEX>
SparseMatrix<TYPE, TYPE_INDEX>::~SparseMatrix() {
  delete _values;
  delete _indices;
  delete _indptr;
}
//Manually initialze
template<typename TYPE, typename TYPE_INDEX>
SparseJoinedMatrix<TYPE, TYPE_INDEX>::SparseJoinedMatrix(unsigned int m,
                                 unsigned int n,
                                 SparseType type,
                                 unsigned short hitems, unsigned short vitems,
                                 float sparsity, bool random) {
  assert(sparsity <= 1.0);
  assert(n % vitems == 0);
  assert(m % hitems == 0);

  _rows = m / vitems;
  _cols = n / hitems;
  _type = type;
  _hitems = hitems;
  _vitems = vitems;
  _sparsity = sparsity;

  unsigned long long size = m * n;
  unsigned long long nnz = (int)(size * sparsity);
  nnz = (int)(nnz / (hitems * vitems)) * (hitems * vitems);
  int alignment = hitems * vitems;
  unsigned int num_indices = nnz / (hitems * vitems); //num of buckets

  unsigned int joined_size = nnz * 2;
/*   unsigned int joined_size = nnz * \
       (sizeof(TYPE) + sizeof(TYPE_INDEX)) * (hitems * vitems); */
  _data = new Blob<TYPE_INDEX>(joined_size, alignment);
  _indptr = new Blob<unsigned int>(_rows + 1);

#if defined(SCATTER)
    _rowind = new Blob<TYPE_INDEX>(m);
    TYPE_INDEX* rowind = _rowind->get();
    for (unsigned i = 0; i < _rows; i++) {
      rowind[i] = i;
    }
#endif

  // randomly initialize the indptr array
  unsigned int *indptr = _indptr->get();
  indptr[0] = 0;
  indptr[_rows] = num_indices;
  unsigned long long count = 0;
  for (unsigned i = 1; i < _rows; i++) {
    if (count < num_indices) {
      int num = (int)(_cols * sparsity);
      if (count + num > num_indices) {
        count = num_indices;
      } else {
        count += num;
      }
    }
    indptr[i] = count;
  }

  TYPE_INDEX *joined = _data->get();
  unsigned chunk = vitems * hitems;
  for (unsigned i = 0; i < _rows; i++) {
    unsigned start = indptr[i];
    unsigned end = indptr[i+1];
    unsigned count = 0;
    for (unsigned j = start; j < end; j++) {
      for (unsigned k = 0; k < chunk; k++) {
        TYPE value = 2.5*k;
        joined[j * 2 * chunk + k] = reinterpret_cast<TYPE_INDEX &>(value);
           //value
        joined[j * 2 * chunk + chunk + k] = count % chunk; //offset
        count++;
      }
    }
  }

  flush_cache(2 * 1024 * 1024, 16);
  //std::cout<<"******** no flush cache *********" <<std::endl;
  // std::cout<<"indptr:"<<std::endl;
  // for (unsigned i = 0; i <= _rows; i++) {
  //   std::cout << indptr[i] <<"\t";//<< std::endl;
  // }
  // std::cout<<"\n\nindices:"<<std::endl;
  // for (unsigned i = 0; i < nnz*2; i++) {
  //   std::cout <<"joined["<<i<<"]="<< joined[i] <<"\t";//<< std::endl;
  // }


  if (random) {
    // not implemented
    assert(false);
  }
  std::cout << "Init joined sparse matrix of (" << m << ", " << n << ")\n";
  std::cout << "\trow: " << _rows << ", cols: " << _cols;
  std::cout << "\ttype: " << (_type == SPARSE_IRREGULAR ? "irregular" :
                             _type == SPARSE_BLOCK ? "block" :
                             _type == SPARSE_BUCKET ? "bucket" :
                             "unknown") << std::endl;
  std::cout << "\thitems: " << _hitems << ", vitems: " << _vitems << std::endl;
  std::cout << "\tsparsity: " << _sparsity << std::endl;

}


//Initialize from real parameters
template<typename TYPE, typename TYPE_INDEX>
SparseJoinedMatrix<TYPE, TYPE_INDEX>::SparseJoinedMatrix(unsigned int m,
                                 unsigned int n,
                                 SparseType type,
                                 unsigned short hitems, unsigned short vitems,
                                 bool random,
                                 std::string ptr_file,
                                 std::string ptr_cmf_file,
                                 std::string v_file) {
  assert(n % vitems == 0);
  assert(m % hitems == 0);

  _rows = m / vitems;
  _cols = n / hitems;
  _type = type;
  _hitems = hitems;
  _vitems = vitems;


  //Read from file
  unsigned int ptr_num = _rows + 1;

  clock_t time_start = clock();
  FILE  *fptr_cmf;
  FILE  *fv;
  fptr_cmf = fopen(ptr_cmf_file.c_str() ,"rt+");
  std::cout << ptr_cmf_file.c_str() << std::endl;
  std::cout << v_file.c_str() << std::endl;
  unsigned int *ptr_cmf = (unsigned int *)calloc(ptr_num,
                                                 sizeof(unsigned int));

  unsigned int shape[2] = {0, 0};
  unsigned int num1 = 0, num2 = 0;
  unsigned int ptr_cnt = 0, v_cnt = 0, idx_cnt = 0;

  fscanf(fptr_cmf, "%d ", &shape[0]);
  fscanf(fptr_cmf, "%d\n", &shape[1]);
  std::cout << "b_shape: " << shape[0]
            << ", " << shape[1] << std::endl;
  assert(shape[0] == m);
  assert(shape[1] == n);
  fscanf(fptr_cmf, "%d\n", &num1);
  fscanf(fptr_cmf, "%d\n", &num2);

  ptr_cmf[0] = 0;
  ptr_cnt++;
  // for (unsigned int i = 1; i <= ptr_num-1; i++){
  while (!feof(fptr_cmf)){
    fscanf(fptr_cmf, "%d ", &ptr_cmf[ptr_cnt]);
    ptr_cnt++;
  }
  fclose(fptr_cmf);
  std::cout << "Read ptr_cmf_file finished. size: " << ptr_cnt << ". ";
  clock_t time_end = clock();
  std::cout << "Time use: "
            << 1000*(time_end-time_start)/(double)CLOCKS_PER_SEC
            << "s" << std::endl;

  unsigned int nnz = ptr_cmf[ptr_cnt-1] * hitems * vitems;
  std::cout << "Num of nonzeros: " << nnz << std::endl;

  nnz = (int)(nnz / (hitems * vitems)) * (hitems * vitems);
  // unsigned int num_indices = nnz;
  int alignment = hitems * vitems;
  // num_indices = nnz / (hitems * vitems); //num of buckets


  _sparsity = 1.0 * nnz / (m * n);


  float *values = (float *)calloc(nnz, sizeof(float));
  // float *values_bin = (float *)calloc(nnz, sizeof(float));
  unsigned short *indices = (unsigned short *)calloc(nnz,
                                                     sizeof(unsigned short));
  /*
  unsigned short *indices_bin = (unsigned short *)calloc(nnz,
                                                      sizeof(unsigned short));
  */
  clock_t time_start2 = clock();
  unsigned int i = 0;

  // if binary exit;
  std::string idx_v_cnt_filename = v_file + ".idx_v_cnt.dat";
  std::string idx_bin_filename   = v_file + ".idx_bin.dat";
  std::string v_bin_filename     = v_file + ".v_bin.dat";
  bool exist_bin_file = false;
  // std::cout << idx_v_cnt_filename.c_str() << std::endl;
  // std::cout << idx_bin_filename.c_str() << std::endl;
  // std::cout << v_bin_filename.c_str() << std::endl;
  // std::cout << access(idx_v_cnt_filename.c_str(), 0) << std::endl;
  // std::cout << access(idx_bin_filename.c_str(), 0) << std::endl;
  // std::cout << access(v_bin_filename.c_str(), 0) << std::endl;
  if (access(idx_v_cnt_filename.c_str(), 0) == 0
      && access(idx_bin_filename.c_str(), 0) == 0
      && access(v_bin_filename.c_str(), 0) == 0 ) {
        exist_bin_file = true;
  }
  else {
    exist_bin_file = false;
  }

  unsigned int vector_length = svcnth();
  //if bin file does not exist, generate bin file.
  if (!exist_bin_file) {
    std::cout << "Bin file does not exist, generating ..." << std::endl;
    fv = fopen(v_file.c_str() ,"rt+");
    clock_t time_start_reading = clock();
    clock_t time_end_reading = clock();
    // while(i < 200){
    while(!feof(fv)){
      if (vector_length == 16){
        fscanf(fv, "%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n\
        %hu %hu %hu %hu %hu %hu %hu %hu %hu %hu %hu %hu %hu %hu %hu %hu\n",
            &values[i*16], &values[i*16+1], &values[i*16+2], &values[i*16+3],
            &values[i*16+4], &values[i*16+5], &values[i*16+6], &values[i*16+7],
            &values[i*16+8], &values[i*16+9], &values[i*16+10],
            &values[i*16+11],
            &values[i*16+12], &values[i*16+13], &values[i*16+14],
            &values[i*16+15],
            &indices[i*16], &indices[i*16+1], &indices[i*16+2],
            &indices[i*16+3],
            &indices[i*16+4], &indices[i*16+5], &indices[i*16+6],
            &indices[i*16+7],
            &indices[i*16+8], &indices[i*16+9], &indices[i*16+10],
            &indices[i*16+11],
            &indices[i*16+12], &indices[i*16+13], &indices[i*16+14],
            &indices[i*16+15]);

        v_cnt += 16;
        idx_cnt += 16;

      }
      else if (vector_length == 8){
        fscanf(fv, "%f %f %f %f %f %f %f %f\n\
        %hu %hu %hu %hu %hu %hu %hu %hu\n",
            &values[i*8], &values[i*8+1], &values[i*8+2], &values[i*8+3],
            &values[i*8+4], &values[i*8+5], &values[i*8+6], &values[i*8+7],
            &indices[i*8], &indices[i*8+1], &indices[i*8+2], &indices[i*8+3],
            &indices[i*8+4], &indices[i*8+5], &indices[i*8+6],
            &indices[i*8+7]);

        v_cnt += 8;
        idx_cnt += 8;
      }


      if(i % 100 == 0){
        std::cout << i*2 << std::endl;
        time_start_reading = time_end_reading;
        time_end_reading = clock();
        std::cout << "time use: "
                  << 1000*(time_end_reading-time_start_reading)/
                    (double)CLOCKS_PER_SEC << "s" << std::endl;
      }
      i ++;
    }
    fclose(fv);

    // save v_cnt and idx_cnt
    std::ofstream cntoutFile(idx_v_cnt_filename);
    cntoutFile << idx_cnt << std::endl;
    cntoutFile.close();

    // save array as bin file
    std::ofstream idxoutFile(idx_bin_filename,
                             std::ios::out | std::ios::binary);
    for (unsigned int i = 0; i < idx_cnt; i++){
        idxoutFile.write((char*)&indices[i], sizeof(unsigned short));
    }
    idxoutFile.close();
    std::cout << "save idx_array as bin file" << std::endl;

    // save array as bin file
    std::ofstream voutFile(v_bin_filename, std::ios::out | std::ios::binary);
    for (unsigned int i = 0; i < v_cnt; i++){
        voutFile.write((char*)&values[i], sizeof(float));
    }
    voutFile.close();
    std::cout << "save v_array as bin file" << std::endl;
  }
  else {
    std::cout << "Loading from bin file!" << std::endl;
    // save v_cnt and idx_cnt
    FILE  *f_idx_v_cnt = fopen(idx_v_cnt_filename.c_str() ,"rt+");
    fscanf(f_idx_v_cnt, "%d", &idx_cnt);
    fclose(f_idx_v_cnt);
    v_cnt = idx_cnt;
    std::cout << "idx_cnt: " << idx_cnt << std::endl;
    std::cout << "v_cnt: " << v_cnt << std::endl;

    clock_t time_start_bin = clock();
    // read idx_array from bin file
    std::ifstream idxinFile(idx_bin_filename, std::ios::in | std::ios::binary);
    if(!idxinFile) {
        std::cout << "bin file not exist" << std::endl;
        assert(0);
    }
    int t = 0;
    while(idxinFile.read((char*)&indices[t], sizeof(unsigned short))) {
      t++;
    }
    idxinFile.close();

    // read v_array from bin file
    std::ifstream vinFile(v_bin_filename, std::ios::in | std::ios::binary);
    if(!vinFile) {
        std::cout << "bin file not exist" << std::endl;
        assert(0);
    }
    t = 0;
    while(vinFile.read((char*)&values[t], sizeof(float))) {
      t++;
    }
    vinFile.close();

    std::cout << "Read v_file finished. size: "
        << v_cnt  << std::endl;
    std::cout << "Read idx_file finished. size: "
        << idx_cnt  << std::endl;
    clock_t time_end2 = clock();
    std::cout << "time use: "
              << 1000*(time_end2-time_start2)/(double)CLOCKS_PER_SEC
              << "s" << std::endl;


  }

  // unsigned int joined_size = nnz * 2;
/*   unsigned int joined_size = nnz * \
       (sizeof(TYPE) + sizeof(TYPE_INDEX)) * (hitems * vitems); */
  // _data = new Blob<TYPE_INDEX>(joined_size, alignment);
  _data = new Blob<TYPE_INDEX>(idx_cnt * 2, alignment);
  _indptr = new Blob<unsigned int>(_rows + 1);

  assert(_rows + 1 == ptr_cnt);
  _data->init_joined_from_array(values, indices, idx_cnt);
  _indptr->init_from_array(ptr_cmf, ptr_cnt);

  flush_cache(2 * 1024 * 1024, 16);

  if (random) {
    // not implemented
    assert(false);
  }
  std::cout << "Init joined sparse matrix of (" << m << ", " << n << ")\n";
  std::cout << "\trow: " << _rows << ", cols: " << _cols;
  std::cout << "\ttype: " << (_type == SPARSE_IRREGULAR ? "irregular" :
                             _type == SPARSE_BLOCK ? "block" :
                             _type == SPARSE_BUCKET ? "bucket" :
                             "unknown") << std::endl;
  std::cout << "\thitems: " << _hitems << ", vitems: " << _vitems << std::endl;
  std::cout << "\tsparsity: " << _sparsity << std::endl;

}

template<typename TYPE, typename TYPE_INDEX>
SparseJoinedMatrix<TYPE, TYPE_INDEX>::~SparseJoinedMatrix() {
  delete _data;
  delete _indptr;
}

template<class TYPE>
TYPE cal_sum_vec(TYPE *c, unsigned int N){
  TYPE sum = 0;
  for(unsigned int i=0; i<N; i++){
    sum += c[i];
  }
  std::cout <<" vec sum = "<< sum << std::endl;
}

#endif
