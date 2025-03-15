CXX=clang++
LLD=lld
CXXFLAGS=-std=c++17 -O3 -fno-math-errno -Wall -flto -I../../cpp
LDFLAGS=-fuse-ld=$(LLD) -O3 -fno-math-errno -flto -Wl,--load-pass-plugin=$(LLDENZYME) -lm

ENZYME_LIB?=/gradbench/enzyme-build/Enzyme/
LLDENZYME=$(ENZYME_LIB)/LLDEnzyme-19.so

EXECUTABLES=run_hello run_gmm run_lstm run_ba run_ht run_kmeans run_particle

all: $(EXECUTABLES)

run_%: run_%.cpp
	$(CXX) -o $@ $^ $(LDFLAGS) $(CXXFLAGS)

clean:
	rm -f $(UTIL_OBJECTS) $(HT_OBJECTS) $(EXECUTABLES)
