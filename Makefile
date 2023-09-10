## Compilers and Flags
CC := CC 
#CC := mpic++
CFLAGS := -g -w -std=c++20  -O3 -march=native -Wno-unknown-pragmas -Wno-deprecated-enum-enum-conversion 
#CFLAGS := -g -w -std=c++20  -O3 -march=native  #nvc++
#CFLAGS := -g -std=c++20  -O0 -l -Wno-unknown-pragmas -Wno-deprecated-enum-enum-conversion -fno-inline -pg #profiling
#CFLAGS := -g -Wall -Wextra -std=c++17 -O3 -Wno-unknown-pragmas 
#CFLAGS := -g -Wall -std=c++17 -Ofast -xHost -xHost -qopt-report-phase=vec,loop -qopt-report=5 
LIB := -Lbuild/ -ltbb 
EIGEN=-I/lustre/home/br-cward/repos/eigen
INC := -Iinclude/ $(EIGEN)


# Nvidia compiler 22.9 only supports up to C++17 - range support was added in C++20
# std::execution port: https://github.com/nvidia/stdexec
# std::mdspan port: https://github.com/kokkos/mdspan
# std::ranges port: https://github.com/ericniebler/range-v3.git

# Set the include directories
INCLUDE_DIRS := $(HOME)/repos/range-v3/ 

# Find all subdirectories of the include directories
SUBDIRS := $(shell find $(INCLUDE_DIRS) -type d)

# Generate the necessary -I flags for all the subdirectories
INCLUDES += $(addprefix -I,$(SUBDIRS))



## Directories
SRC := src
TESTS := tests
EXE := bin/minicombust
TEST_EXE := bin/minicombust_tests


ifdef PAPI
	INC += -DPAPI -I/opt/cray/pe/papi/6.0.0.7/include
	LIB += -L/opt/cray/pe/papi/6.0.0.7/lib64 -lpapi -lpfm
endif

SOURCES := $(shell find $(SRC) -type f -name *.c -o -name *.cpp ! -name minicombust.cpp)
OBJECTS := $(patsubst $(SRC)/%,build/%,$(SOURCES:.cpp=.o))

all: $(EXE) $(TEST_EXE)

notest: $(EXE)

$(EXE): $(OBJECTS)
	$(CC) $(CFLAGS) $(INC) $(SRC)/minicombust.cpp -c -o build/minicombust.o 
	@echo ""
	@echo "Linking..."
	$(CC) $(LIB) $^ build/minicombust.o -o $(EXE) 

$(TEST_EXE): $(OBJECTS)
	$(CC) $(CFLAGS) $(INC) $(SRC)/minicombust.cpp -c -o build/minicombust.o 
	$(CC) $(CFLAGS) $(INC) $(TESTS)/minicombust_tests.cpp -c -o build/minicombust_tests.o 
	@echo ""
	@echo "Linking..."
	$(CC) $(LIB) $^ build/minicombust.o -o $(EXE) 
	$(CC) $(LIB) $^ build/minicombust_tests.o -o $(TEST_EXE)

build/%.o: $(SRC)/%.cpp
	@mkdir -p bin build out $(dir $@)
	$(CC) $(CFLAGS) $(INC) $< -c -o $@ 

clean:
	@echo "Cleaning..."
	rm -rf build/* $(EXE)
	@echo ""
	rm output.txt error.txt callgrind.out.*

cleant:
	rm output.txt error.txt callgrind.out.*

.PHONY: clean
