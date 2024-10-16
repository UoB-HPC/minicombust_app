## Compilers and Flags
CC := mpicxx
#-G
CFLAGS := -fopenmp -g -Wall -Wextra -std=c++2a -O3 -march=native -Wno-unknown-pragmas -Wno-deprecated-enum-enum-conversion

LIB := -Lbuild/
INC := -Iinclude/

## Directories
SRC := src
TESTS := tests
EXE := bin/minicombust
EXE_GPU := bin/gpu_minicombust
TEST_EXE := bin/minicombust_tests
CPX_EXE := bin/libminicombust.a

ifdef CPX_INSTALL_PATH
	INC += -I$(CPX_INSTALL_PATH)
endif

ifdef MPI_INSTALL_PATH
	INC += -I$(MPI_INSTALL_PATH)/include
	LIB += -L$(MPI_INSTALL_PATH)/lib -lmpi
endif

ifdef CUDA_INSTALL_PATH
	CC := nvcc
	CFLAGS := --expt-relaxed-constexpr -rdynamic --expt-extended-lambda -gencode arch=compute_90,code=sm_90 -fopenmp --generate-line-info  -pg -g -forward-unknown-to-host-compiler -Xcompiler -std=c++2a -O3 -Wno-unknown-pragmas -Wno-deprecated-enum-enum-conversion --expt-relaxed-constexpr --expt-extended-lambda
	NVCC := nvcc
	NVFLAGS := --expt-relaxed-constexpr --expt-extended-lambda -forward-unknown-to-host-compiler -rdynamic -fopenmp --extended-lambda -pg -g -O3 -gencode arch=compute_90,code=sm_90 
	INC += -I$(CUDA_INSTALL_PATH)/include
	LIB += -L$(CUDA_INSTALL_PATH)/lib64 -lcudart -lnvToolsExt
endif

ifdef PETSC_INSTALL_PATH
	INC += -I$(PETSC_INSTALL_PATH)/include
	LIB += -L$(PETSC_INSTALL_PATH)/lib -lpetsc
endif

ifdef AMGX_INSTALL_PATH
	INC += -I$(AMGX_INSTALL_PATH)/include
	LIB += -L$(AMGX_INSTALL_PATH)/lib -lamgxsh -lamgx
endif

ifdef PAPI
	INC += -DPAPI -I/opt/cray/pe/papi/6.0.0.7/include
	LIB += -L/opt/cray/pe/papi/6.0.0.7/lib64 -lpapi -lpfm
endif

SOURCES := $(shell find $(SRC) -type f -name *.c -o -name *.cpp ! -name minicombust*.cpp)
OBJECTS := $(patsubst $(SRC)/%,build/%,$(SOURCES:.cpp=.o))

all: $(EXE) $(TEST_EXE)

gpu: $(EXE_GPU)

notest: $(EXE)

cpx: $(CPX_EXE)

$(EXE): $(OBJECTS)
	$(CC) $(CFLAGS) $(INC) $(SRC)/minicombust.cpp -c -o build/minicombust.o 
	@echo ""
	@echo "Linking..."
	$(CC) ${LIB} $^ build/minicombust.o -o $(EXE)

$(EXE_GPU): $(OBJECTS) build/gpu_kernels.o
	$(CC) $(CFLAGS) $(INC) $(SRC)/minicombust.cpp -Dhave_gpu -c -o build/minicombust.o
	@echo ""
	@echo "Linking..."
	$(NVCC) -forward-unknown-to-host-compiler -fopenmp $(LIB) $^ build/minicombust.o -Dhave_gpu -o $(EXE_GPU)

$(CPX_EXE) : $(OBJECTS) build/gpu_kernels.o
	$(CC) $(CFLAGS) $(INC) $(SRC)/minicombust_cpx.cpp -Dhave_gpu -Ddefgpucom -c -o build/minicombust_cpx.o
	$(AR) rcs $(CPX_EXE) $^ build/minicombust_cpx.o build/gpu_kernels.o

build/gpu_kernels.o: include/flow/gpu/gpu_kernels.cu
	$(NVCC) $(NVFLAGS) $(INC) include/flow/gpu/gpu_kernels.cu -c -o $@

build/%.o: $(SRC)/%.cpp
	@mkdir -p bin build out $(dir $@)
	$(CC) $(CFLAGS) $(INC) $< -c -o $@ 

clean:
	@echo "Cleaning..."
	rm -rf build/* #$(EXE)
	@echo ""

.PHONY: clean
