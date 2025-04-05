CXX?=c++
CC?=cc
CXXFLAGS=-std=c++17 -O2 -Wall -fopenmp -I../../cpp
LDFLAGS=-lm -ladept -fopenmp

EXECUTABLES=hello gmm ht ba lstm ode llsq det

all: $(EXECUTABLES)

%: %.cpp
	$(CXX) -o $@ $^ $(LDFLAGS) $(CXXFLAGS)

clean:
	rm -f $(EXECUTABLES)
