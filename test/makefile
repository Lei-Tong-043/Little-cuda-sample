# compiler
NVCC = /online1/public/support/amd/cuda/12.2/bin/nvcc
CC   = /online1/public/support/intel/intel_compiler/2024u1/compiler/2024.1/bin/icx
CXX  = /online1/public/support/intel/intel_compiler/2024u1/compiler/2024.1/bin/icpx

C_FLAGS   = -fopenmp
CXX_FLAGS = $(C_FLAGS)

# cuda include && lib
CUDA_LIB = -L/online1/public/support/amd/cuda/12.2/lib64 -lcudart
CUDA_INC = -I/online1/public/support/amd/cuda/12.2/include 

# CUDA architecture for 4090
CUDA_ARCH = -gencode arch=compute_89,code=sm_89

# source and object files
CUDA_SRC = test_device.cu
CUDA_OBJ = test_device.o
CPP_SRC  = test_host.cpp
CPP_OBJ  = test_host.o

# target exec file
TARGET = test

# default rule
all: $(TARGET)

# linking rule
$(TARGET): $(CPP_OBJ) $(CUDA_OBJ)
	$(CXX) -o $@ $(CPP_OBJ) $(CUDA_OBJ) $(CUDA_LIB)

# rule to compile .cpp files
%.o: %.cpp
	$(CXX) -c $< $(CUDA_INC) -o $@

# rule to compile .cu files
%.o: %.cu
	$(NVCC) -c $< $(CUDA_INC) $(CUDA_ARCH) -o $@

# clean rule
.PHONY: clean
clean:
	rm -f $(TARGET) $(CPP_OBJ) $(CUDA_OBJ)
