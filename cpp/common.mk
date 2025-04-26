# Common Makefile definitions that are used by many of the C++-based
# tools. You do not have to use it, but the cpp.py tooling expects an
# interface that behaves like this.
#
# The EXECUTABLES variable (and the 'all' target) is not needed for
# cpp.py, but only for local testing.

CXX?=c++
CC?=cc
CXXFLAGS?=-std=c++17 -O3 -march=native -Wall -I../../cpp
LDFLAGS?=-lm

all: $(addprefix bin/, $(EXECUTABLES))

bin/%: %.cpp $(EXTRA_DEPS)
	@mkdir -p bin
	$(CXX) -o $@ $^ $(LDFLAGS) $(CXXFLAGS)

clean:
	rm -f bin *.o
