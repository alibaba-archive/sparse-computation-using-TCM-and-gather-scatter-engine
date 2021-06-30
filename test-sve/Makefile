
ASM=asm
OBJ=obj
BIN=bin
SRC=src
DIS=dis
CC=aarch64-linux-gnu-g++ # aarch64-linux-gnu-g++
OBJDUMP=aarch64-linux-gnu-objdump # aarch64-linux-gnu-objdump
CFLAGS=-O2 -g -Wall -march=armv8.2-a+sve+simd --static -ftree-vectorizer-verbose=2 -msve-vector-bits=128
CCFLAGS=CFLAGS
DEPS=mm.h

.PRECIOUS: $(ASM)/%.s

$(ASM)/%.s: $(SRC)/%.cpp
	$(CC) $(CCFLAGS) -S -o $@ $<

$(OBJ)/%.o: $(ASM)/%.s
	$(CC) $(CCFLAGS) -c -o $@ $<
	$(OBJDUMP) -d $@ > $(DIS)/$(@F).dis

$(SRC)/dense_split.cpp : $(SRC)/dense_split.py
	rm -rf $@
	python $< --m 128 --n 128 --k 128 --mm 1 --mi 4 --ni 32 --vl 8 > $@
	python $< --m 1 --n 1024 --k 1024 --mm 1 --mi 1 --ni 8 --vl 8 >> $@
#	python $< --m 1 --n 1024 --k 1024 --mm 1 --mi 1 --ni 8 --vl 8 --fp16 >> $@
#	python $< --m 1 --n 1024 --k 1024 --mm 1 --mi 1 --ni 8 --vl 8 --fp32 >> $@

$(SRC)/dense_conv_split.cpp : $(SRC)/dense_conv_split.py
	rm -rf $@
	python $< --co_i 1 --ci_i 32 --kh_i 1 --kw_i 3 --vl 8 --fp16 >> $@
	# python $< --co_i 1 --ci_i 32 --kh_i 1 --kw_i 3 --vl 8 --fp32 >> $@
	# python $< --co_i 1 --ci_i 8 --kh_i 2 --kw_i 2 --vl 8 --fp32 >> $@

$(SRC)/sparse_split.cpp : $(SRC)/sparse_split.py
	rm -rf $@
	python $< --m 128 --n 128 --k 128 --mm 8 --mi 8 --vl 8 > $@

$(SRC)/sparse_split_transposed.cpp : $(SRC)/sparse_split.py
	rm -rf $@
	python $< --m 128 --n 128 --k 128 --mm 16 --mi 8 --vl 8 --trans > $@

$(SRC)/sparse_conv_split.cpp : $(SRC)/sparse_conv_split.py
	rm -rf $@
	python $< --sp_unroll 8 --vl 8 > $@

$(BIN)/dense_harness: $(OBJ)/dense_harness.o $(OBJ)/dense.o $(OBJ)/dense_split.o $(OBJ)/utils.o $(OBJ)/dense_conv_split.o
	$(CC) $(CFLAGS) -o $@ $^

$(BIN)/sparse_harness: $(OBJ)/sparse_harness.o $(OBJ)/sparse.o $(OBJ)/utils.o $(OBJ)/sparse_split.o $(OBJ)/sparse_split_transposed.o $(OBJ)/sparse_conv_split.o
	$(CC) $(CFLAGS) -o $@ $^

$(BIN)/testbench: $(OBJ)/testbench.o $(OBJ)/tb_kernels.o $(OBJ)/utils.o $(OBJ)/sparse.o  $(OBJ)/sparse_split_transposed.o $(OBJ)/sparse_conv_split.o
	$(CC) $(CFLAGS) -o $@ $^

$(BIN)/testbench_convfp16: $(OBJ)/testbench_convfp16.o $(OBJ)/tb_kernels.o $(OBJ)/utils.o $(OBJ)/sparse.o  $(OBJ)/sparse_split_transposed.o $(OBJ)/sparse_conv_split.o
	$(CC) $(CFLAGS) -o $@ $^

$(BIN)/testbench_dense: $(OBJ)/testbench_dense.o $(OBJ)/tb_kernels.o $(OBJ)/utils.o $(OBJ)/dense.o $(OBJ)/dense_split.o $(OBJ)/utils.o $(OBJ)/dense_conv_split.o
	$(CC) $(CFLAGS) -o $@ $^

$(BIN)/testbench_spmv16: $(OBJ)/testbench_spmv16.o $(OBJ)/tb_kernels.o $(OBJ)/utils.o $(OBJ)/sparse.o  $(OBJ)/sparse_split_transposed.o $(OBJ)/sparse_conv_split.o
	$(CC) $(CFLAGS) -o $@ $^

$(BIN)/testbench_spmv32: $(OBJ)/testbench_spmv32.o $(OBJ)/tb_kernels.o $(OBJ)/utils.o $(OBJ)/sparse.o  $(OBJ)/sparse_split_transposed.o $(OBJ)/sparse_conv_split.o
	$(CC) $(CFLAGS) -o $@ $^

dense : dense_sve_unroll

#### Dense kernel ###
dense_sve_unroll : CCFLAGS = -DHGEMM_UNROLLED -DDIM_N=1024 -DDIM_K=1024 $(CFLAGS)
dense_sve_unroll : $(BIN)/dense_harness

dense_sve_unroll_split : CCFLAGS = -DDENSE_FP16 -DHGEMM_UNROLLED_SPLIT $(CFLAGS)
dense_sve_unroll_split : $(BIN)/dense_harness

dense_sve : CCFLAGS = -DHGEMM -DDIM_N=1024 -DDIM_K=1024 $(CFLAGS)
dense_sve : $(BIN)/dense_harness

dense_scalar : CCFLAGS = -DSCALAR $(CFLAGS)
dense_scalar : $(BIN)/dense_harness

dense_scalar_transposed : CCFLAGS = -DSCALAR_TRANSPOSED $(CFLAGS)
dense_scalar_transposed : $(BIN)/dense_harness

dense_vector_sve : CCFLAGS = -DVECTOR -DDENSE_FP16 -DDIM_M=1 -DDIM_N=1024 -DDIM_K=1024 $(CFLAGS)
dense_vector_sve : $(BIN)/dense_harness

mm_sve_split_fp16_fp32 : CCFLAGS = -DVECTOR -DDENSE_FP16 -DCOMPUTE_FP32 -DDIM_M=1 -DDIM_N=4096 -DDIM_K=1024 $(CFLAGS)
mm_sve_split_fp16_fp32 : $(BIN)/dense_harness

dense_conv_fp16_f32_compute : CCFLAGS = -DDENSE_FP16 -DCONV -DCOMPUTE_FP32 $(CFLAGS)
dense_conv_fp16_f32_compute : $(BIN)/dense_harness

dense_conv_fp16 : CCFLAGS = -DDENSE_FP16 -DCONV $(CFLAGS)
dense_conv_fp16 : $(BIN)/dense_harness

#### Sparse kernel ####
sparse_scalar : CCFLAGS = -DSCALAR $(CFLAGS)
sparse_scalar : $(BIN)/sparse_harness


#----------------spmv block ---------------#

sparse_block : CCFLAGS = -DSBLOCK $(CFLAGS)
sparse_block : $(BIN)/sparse_harness

sparse_block_vertical_split : CCFLAGS = -DSPLIT -DSBLOCK $(CFLAGS)
sparse_block_vertical_split : $(BIN)/sparse_harness $(SRC)/sparse_split.py
	rm -rf $(SRC)/sparse_split.cpp

sparse_block_vertical_split_trans : CCFLAGS = -DSPLIT -DTRANSPOSED -DSBLOCK $(CFLAGS)
sparse_block_vertical_split_trans : $(BIN)/sparse_harness $(SRC)/sparse_split.py
	rm -rf $(SRC)/sparse_split_transposed.cpp

spmv_block: CCFLAGS = -DSBLOCK -DDIM_M=1 -DDIM_N=1024 -DDIM_K=1024 $(CFLAGS)
spmv_block : $(BIN)/sparse_harness

spmv_block_fp32 : CCFLAGS = -DSBLOCK -DDIM_M=1 -DDIM_N=1024 -DDIM_K=1024 -DFP32 $(CFLAGS) #HL: FP32 version
spmv_block_fp32 : $(BIN)/sparse_harness

spmv_block_split: CCFLAGS = -DSBLOCK -DSPLIT -DDIM_M=1 -DDIM_N=1024 -DDIM_K=1024 $(CFLAGS)
spmv_block_split : $(BIN)/sparse_harness

spmv_block_fp16: CCFLAGS = -DSBLOCK -DDENSE_FP16 -DDIM_M=1 -DDIM_N=1024 -DDIM_K=1024 $(CFLAGS)
spmv_block_fp16 : $(BIN)/sparse_harness

spmv_block_vertical_fp32: CCFLAGS = -DSBLOCK -DCOMPUTE_FP32 -DDENSE_FP16 -DDIM_M=1 -DDIM_N=1024 -DDIM_K=1024 $(CFLAGS)
spmv_block_vertical_fp32 : $(BIN)/sparse_harness

spmv_block_horizonal_fp16 : CCFLAGS = -DSBLOCK -DDENSE_FP16 -DHORIZON -DDIM_M=1 -DDIM_N=1024 -DDIM_K=1024 $(CFLAGS)
spmv_block_horizonal_fp16 : $(BIN)/sparse_harness

spmv_block_horizonal_fp32 : CCFLAGS = -DSBLOCK -DCOMPUTE_FP32 -DDENSE_FP16 -DHORIZON -DDIM_M=1 -DDIM_N=1024 -DDIM_K=1024 $(CFLAGS)
spmv_block_horizonal_fp32 : $(BIN)/sparse_harness

sp_conv_block_vertical_fp16_fp32_compute : CCFLAGS = -DSBLOCK -DCOMPUTE_FP32 -DDENSE_FP16 -DCONV  -DDIM_M=1 $(CFLAGS)
sp_conv_block_vertical_fp16_fp32_compute : $(BIN)/sparse_harness

sp_conv_block_horizontal_fp16_fp32_compute : CCFLAGS = -DSBLOCK -DCOMPUTE_FP32 -DDENSE_FP16 -DCONV -DHORIZON -DDIM_M=1 $(CFLAGS)
sp_conv_block_horizontal_fp16_fp32_compute : $(BIN)/sparse_harness

sp_conv_block_vertical_fp16 : CCFLAGS = -DSBLOCK -DDENSE_FP16 -DCONV  -DDIM_M=1 $(CFLAGS)
sp_conv_block_vertical_fp16 : $(BIN)/sparse_harness

sp_conv_block_horizontal_fp16 : CCFLAGS = -DSBLOCK -DDENSE_FP16 -DCONV -DHORIZON -DDIM_M=1 $(CFLAGS)
sp_conv_block_horizontal_fp16 : $(BIN)/sparse_harness

#----------------spmv bucket ---------------#

spmv_bucket : CCFLAGS = -DSBUCKET -DDIM_M=1 -DDIM_N=8 -DDIM_K=8 $(CFLAGS)
spmv_bucket : $(BIN)/sparse_harness

spmv_bucket_split : CCFLAGS = -DSBUCKET -DSPLIT -DDIM_M=1 -DDIM_N=1024 -DDIM_K=1024 $(CFLAGS)
spmv_bucket_split : $(BIN)/sparse_harness

spmv_bucket_vertical_fp16 : CCFLAGS = -DSBUCKET -DDENSE_FP16 -DDIM_M=1 -DDIM_N=1024 -DDIM_K=1024 $(CFLAGS)
spmv_bucket_vertical_fp16 : $(BIN)/sparse_harness

spmv_bucket_joined : CCFLAGS = -DSBUCKET -DJOINED -DDIM_M=1 -DDIM_N=8 -DDIM_K=8 $(CFLAGS)
spmv_bucket_joined : $(BIN)/sparse_harness

spmv_bucket_joined_sparsefp32 : CCFLAGS = -DSBUCKET -DJOINED -DDIM_M=1 -DDIM_N=1024 -DDIM_K=1024 -DSPARSE_FP32 $(CFLAGS)
spmv_bucket_joined_sparsefp32 : $(BIN)/sparse_harness

spmv_bucket_joined_densefp16 : CCFLAGS = -DSBUCKET -DJOINED -DDIM_M=1 -DDIM_N=1024 -DDIM_K=1024 -DDENSE_FP16 $(CFLAGS)
spmv_bucket_joined_densefp16 : $(BIN)/sparse_harness

spmv_bucket_joined_densefp16_fused : CCFLAGS = -DSBUCKET -DJOINED -DDIM_M=1 -DDIM_N=1024 -DDIM_K=1024 -DDENSE_FP16 -DFUSED $(CFLAGS)
spmv_bucket_joined_densefp16_fused : $(BIN)/sparse_harness


spmv_bucket_joined_densefp32 : CCFLAGS = -DSBUCKET -DCOMPUTE_FP32 -DJOINED -DDIM_M=1 -DDIM_N=1024 -DDIM_K=1024 -DDENSE_FP16 $(CFLAGS)
spmv_bucket_joined_densefp32 : $(BIN)/sparse_harness

spmv_joined_bucket_vertical_split_fp16 : CCFLAGS = -DSBUCKET -DSPLIT -DJOINED -DDIM_M=1 -DDIM_N=1024 -DDIM_K=1024 -DDENSE_FP16 $(CFLAGS)
spmv_joined_bucket_vertical_split_fp16 : $(BIN)/sparse_harness

conv_bucket_joined_densefp16 : CCFLAGS = -DSBUCKET -DJOINED -DB_H=8 -DB_W=8 -DA_H=8 $(CFLAGS)
conv_bucket_joined_densefp16 : $(BIN)/sparse_harness

spmv_bucket_joined_split : CCFLAGS = -DSBUCKET -DJOINED -DSPLIT -DDIM_M=1 -DDIM_N=1024 -DDIM_K=1024 $(CFLAGS)
spmv_bucket_joined_split : $(BIN)/sparse_harness

conv_horizontal_bucket_joined_sp16 : CCFLAGS = -DSBUCKET -DHORIZON -DJOINED -DCONV -DDENSE_FP16 -DVNUM=1 $(CFLAGS)
conv_horizontal_bucket_joined_sp16 : $(BIN)/sparse_harness

conv_horizontal_bucket_joined_sp32 : CCFLAGS = -DSBUCKET -DCOMPUTE_FP32 -DHORIZON -DJOINED -DCONV -DDENSE_FP16 $(CFLAGS)
conv_horizontal_bucket_joined_sp32 : $(BIN)/sparse_harness

conv_vertical_bucket_joined_sp16 : CCFLAGS = -DSBUCKET -DJOINED -DCONV -DDENSE_FP16 $(CFLAGS)
conv_vertical_bucket_joined_sp16 : $(BIN)/sparse_harness

conv_vertical_bucket_joined_sp32 : CCFLAGS = -DSBUCKET -DCOMPUTE_FP32 -DJOINED -DCONV -DDENSE_FP16 $(CFLAGS)
conv_vertical_bucket_joined_sp32 : $(BIN)/sparse_harness

spmv_joined_bucket_horizontal_fp16 : CCFLAGS = -DSBUCKET -DJOINED -DHORIZON -DDENSE_FP16 -DDIM_M=1 -DDIM_N=1024 -DDIM_K=1024 $(CFLAGS)
spmv_joined_bucket_horizontal_fp16 : $(BIN)/sparse_harness

spmv_joined_bucket_horizontal_fp32 : CCFLAGS = -DSBUCKET -DJOINED -DCOMPUTE_FP32 -DHORIZON -DDENSE_FP16 -DDIM_M=1 -DDIM_N=1024 -DDIM_K=1024 $(CFLAGS)
spmv_joined_bucket_horizontal_fp32 : $(BIN)/sparse_harness

sp_conv_horizontal_fp16_fp32_compute : CCFLAGS = -DSBUCKET -DCOMPUTE_FP32 -DHORIZON -DJOINED -DCONV -DDENSE_FP16 $(CFLAGS)
sp_conv_horizontal_fp16_fp32_compute : $(BIN)/sparse_harness

sp_conv_vertical_fp16_fp32_compute : CCFLAGS = -DSBUCKET -DCOMPUTE_FP32 -DJOINED -DCONV -DDENSE_FP16 $(CFLAGS)
sp_conv_vertical_fp16_fp32_compute : $(BIN)/sparse_harness

sp_conv_horizontal_fp16 : CCFLAGS = -DSBUCKET -DHORIZON -DJOINED -DCONV -DDENSE_FP16 $(CFLAGS)
sp_conv_horizontal_fp16 : $(BIN)/sparse_harness

sp_conv_vertical_fp16 : CCFLAGS = -DSBUCKET -DJOINED -DCONV -DDENSE_FP16 $(CFLAGS)
sp_conv_vertical_fp16 : $(BIN)/sparse_harness

sp_conv_vertical_fp16_fused : CCFLAGS = -DSBUCKET -DJOINED -DCONV -DDENSE_FP16 -DFUSED $(CFLAGS)
sp_conv_vertical_fp16_fused : $(BIN)/sparse_harness

spmv_joined_bucket_vertical_scatter_fp16 : CCFLAGS = -DSCATTER -DSBUCKET -DJOINED -DDIM_M=1 -DDIM_N=1024 -DDIM_K=1024 -DDENSE_FP16 $(CFLAGS)
spmv_joined_bucket_vertical_scatter_fp16 : $(BIN)/sparse_harness


#--------------------Test bench kernels----------------------#

spmv_vertical_correct_bucket_fp16 : CCFLAGS = -DV_MODE -DDIM_N=8 -DDIM_K=8 -DA_H=8 -DA_W=1 -DSBUCKET $(CFLAGS)
spmv_vertical_correct_bucket_fp16 :  $(BIN)/testbench_spmv16

spmv_horizontal_correct_bucket_fp16 : CCFLAGS = -DH_MODE -DB_H=8 -DB_W=8 -DA_H=8 -DA_W=1 -DSBUCKET $(CFLAGS)
spmv_horizontal_correct_bucket_fp16 :  $(BIN)/testbench_spmv16

spmv_vertical_correct_block_fp16 : CCFLAGS = -DV_MODE -DDIM_N=8 -DDIM_K=8 -DA_H=8 -DA_W=1 -DSBLOCK $(CFLAGS)
spmv_vertical_correct_block_fp16 :  $(BIN)/testbench_spmv16

spmv_horizontal_correct_block_fp16 : CCFLAGS = -DH_MODE -DB_H=8 -DB_W=8 -DA_H=8 -DA_W=1 -DSBLOCK $(CFLAGS)
spmv_horizontal_correct_block_fp16 :  $(BIN)/testbench_spmv16

spmv_vertical_correct_bucket_fp32 : CCFLAGS = -DV_MODE -DDIM_N=8 -DDIM_K=8 -DA_H=8 -DA_W=1 -DSBUCKET -DCOMPUTE_FP32=1 $(CFLAGS)
spmv_vertical_correct_bucket_fp32 :  $(BIN)/testbench_spmv32

spmv_horizontal_correct_bucket_fp32 : CCFLAGS = -DH_MODE -DB_H=8 -DB_W=8 -DA_H=8 -DA_W=1 -DSBUCKET -DCOMPUTE_FP32=1 $(CFLAGS)
spmv_horizontal_correct_bucket_fp32 :  $(BIN)/testbench_spmv32

spmv_vertical_correct_block_fp32 : CCFLAGS = -DV_MODE -DDIM_N=8 -DDIM_K=8 -DA_H=8 -DA_W=1 -DSBLOCK -DCOMPUTE_FP32=1 $(CFLAGS)
spmv_vertical_correct_block_fp32 :  $(BIN)/testbench_spmv32

spmv_horizontal_correct_block_fp32 : CCFLAGS = -DH_MODE -DB_H=8 -DB_W=8 -DA_H=8 -DA_W=1 -DSBLOCK -DCOMPUTE_FP32=1 $(CFLAGS)
spmv_horizontal_correct_block_fp32 :  $(BIN)/testbench_spmv32

sp_conv_vertical_fp16_fp32_compute_ref : CCFLAGS = -DV_MODE -DCONV -DSBUCKET $(CFLAGS)
sp_conv_vertical_fp16_fp32_compute_ref : $(BIN)/testbench_convfp16

sp_conv_horizontal_fp16_fp32_compute_ref : CCFLAGS = -DH_MODE -DCONV -DSBUCKET $(CFLAGS)
sp_conv_horizontal_fp16_fp32_compute_ref : $(BIN)/testbench_convfp16

sp_conv_block_vertical_fp16_fp32_compute_ref : CCFLAGS = -DV_MODE -DCONV -DSBLOCK $(CFLAGS)
sp_conv_block_vertical_fp16_fp32_compute_ref : $(BIN)/testbench_convfp16

sp_conv_block_horizontal_fp16_fp32_compute_ref : CCFLAGS = -DH_MODE -DCONV -DSBLOCK $(CFLAGS)
sp_conv_block_horizontal_fp16_fp32_compute_ref : $(BIN)/testbench_convfp16

# sp_conv_vertical_fp16_fp32_compute_ref : CCFLAGS = -DV_MODE -DCONV -DSBUCKET $(CFLAGS)
# sp_conv_vertical_fp16_fp32_compute_ref : $(BIN)/testbench

# sp_conv_horizontal_fp16_fp32_compute_ref : CCFLAGS = -DH_MODE -DCONV -DSBUCKET $(CFLAGS)
# sp_conv_horizontal_fp16_fp32_compute_ref : $(BIN)/testbench

# sp_conv_block_vertical_fp16_fp32_compute_ref : CCFLAGS = -DV_MODE -DCONV -DSBLOCK $(CFLAGS)
# sp_conv_block_vertical_fp16_fp32_compute_ref : $(BIN)/testbench

# sp_conv_block_horizontal_fp16_fp32_compute_ref : CCFLAGS = -DH_MODE -DCONV -DSBLOCK $(CFLAGS)
# sp_conv_block_horizontal_fp16_fp32_compute_ref : $(BIN)/testbench


sp_conv_vertical_fp16_fp32_compute_test : CCFLAGS = -DV_MODE -DCONV $(CFLAGS)
sp_conv_vertical_fp16_fp32_compute_test: $(BIN)/testbench

dense_conv_ref : CCFLAGS = -DV_MODE -DCONV $(CFLAGS)
dense_conv_ref : $(BIN)/testbench_dense

testg: $(SRC)/g.cpp
	$(CC) $(CFLAGS) -o $(BIN)/$@ $^

testg_fp16: $(SRC)/g_fp16.cpp
	$(CC) $(CFLAGS) -o $(BIN)/$@ $^

test_conv_sveinst: $(SRC)/test_conv_sveinst.cpp
	$(CC) $(CFLAGS) -o $(BIN)/$@ $^
	
testScatter_fp16: $(SRC)/scatter_fp16.cpp
	$(CC) $(CFLAGS) -o $(BIN)/$@ $^

testScatter_fp32: $(SRC)/scatter_fp32.cpp
	$(CC) $(CFLAGS) -o $(BIN)/$@ $^

testFused: $(SRC)/fused.cpp
	$(CC) $(CFLAGS) -o $(BIN)/$@ $^
	$(OBJDUMP) -d $(BIN)/$@ > $(DIS)/$(@F).dis

testh: $(SRC)/h.cpp
	$(CC) $(CFLAGS) -o $(BIN)/$@ $^

testi: $(SRC)/i.cpp
	$(CC) $(CFLAGS) -o $(BIN)/$@ $^

gemm_eigen: $(SRC)/gemm_eigen.cpp
	$(CC) -I ../../eigen-3.3.7/ -O3 -g -Wall -march=armv8.4-a+simd+sve+fp16fml -static -o $(BIN)/$@ $^
	$(OBJDUMP) -d $(BIN)/$@ > $(DIS)/$(@F).dis
clean :
	rm -rf $(ASM)/*
	rm -rf $(OBJ)/*
	rm -rf $(BIN)/*
	rm -rf $(DIS)/*
	rm -rf $(SRC)/sparse_split.cpp
	rm -rf $(SRC)/dense_split.cpp
	rm -rf $(SRC)/dense_conv_split.cpp
	rm -rf $(SRC)/sparse_split_transposed.cpp
	rm -rf $(SRC)/sparse_conv_split.cpp
