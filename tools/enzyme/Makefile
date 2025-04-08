CXX=clang++
LLD=lld
CXXFLAGS=-std=c++17 -O3 -fno-math-errno -Wall -flto -I../../cpp
LDFLAGS=-fuse-ld=$(LLD) -O3 -fno-math-errno -flto -Wl,--load-pass-plugin=$(LLDENZYME) -lm

ENZYME_LIB?=/gradbench/enzyme-build/Enzyme/
LLDENZYME=$(ENZYME_LIB)/LLDEnzyme-19.so

EXECUTABLES=hello gmm lstm ba ht kmeans particle saddle ode llsq det logsumexp

all: $(EXECUTABLES)

%: %.cpp
	$(CXX) -o $@ $^ $(LDFLAGS) $(CXXFLAGS)

clean:
	rm -f $(EXECUTABLES)
