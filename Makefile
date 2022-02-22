## Compilers and Flags
CC := g++ 
CFLAGS := -g -Wall -std=c++17
LIB := -Lbuild/
INC := -Iinclude/


## Directories
SRC := src
EXE := bin/minicombust



SOURCES := $(shell find $(SRC) -type f -name *.c -o -name *.cpp)
OBJECTS := $(patsubst $(SRC)/%,build/%,$(SOURCES:.cpp=.o))

all: $(EXE)

$(EXE): $(OBJECTS)
	@echo ""
	@echo "Linking..."
	$(CC) $(LIB) $^ -o $(EXE) 

build/%.o: $(SRC)/%.cpp
	@mkdir -p bin build $(dir $@)
	$(CC) $(CFLAGS) $(INC) $< -c -o $@ 

clean:
	@echo "Cleaning..."
	rm -rf build/* $(EXE)
	@echo ""

# Tests
tests:
	$(CC) $(CFLAGS) test/tester.cpp $(INC) $(LIB) -o bin/tester

.PHONY: clean
