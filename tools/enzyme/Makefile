CXX=clang++-19
LLD=lld-19
CXXFLAGS=-std=c++17 -O3 -Wall -flto
LDFLAGS=-fuse-ld=$(LLD) -O3 -flto -Wl,--load-pass-plugin=$(LLDENZYME) -lm

LLDENZYME?=/home/gradbench/enzyme-build/Enzyme/LLDEnzyme-19.so

UTIL_OBJECTS=../../cpp/adbench/shared/utils.o ../../cpp/adbench/io.o
HELLO_OBJECTS=EnzymeHello.o run_hello.o
EXECUTABLES=run_hello

all: $(EXECUTABLES)

run_hello: $(HELLO_OBJECTS) $(UTIL_OBJECTS)
	$(CXX) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $< -c -o $@ $(CXXFLAGS) -I../../cpp

clean:
	rm -f $(UTIL_OBJECTS) $(HELLO_OBJECTS) $(EXECUTABLES)
