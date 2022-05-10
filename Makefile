## Compilers and Flags
CC := icpc 
#CC := CC 
#CFLAGS := -g -Wall -std=c++17 -O3 -march=native -Rpass='(loop-vectorize|inline)' -fsave-optimization-record 
CFLAGS := -g -Wall -std=c++17 -Ofast -xHost -xHost -qopt-report-phase=vec,loop -qopt-report=5 
LIB := -Lbuild/
INC := -Iinclude/


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

$(EXE): $(OBJECTS)
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


.PHONY: clean
