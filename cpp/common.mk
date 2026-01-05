# Common Makefile definitions that are used by many of the C++-based
# tools. You do not have to use it, but the cpp.py tooling expects an
# interface that behaves like this.
#
# The EXECUTABLES variable (and the 'all' target) is not needed for
# cpp.py, but only for local testing.

CXX?=c++
CC?=cc
CXXFLAGS?=-std=c++17 -O3 -Wall -I../../cpp
LDFLAGS?=-lm

MULTITHREADED=no
NATIVE=yes

# This needs to be set to "no" anytime we compile in a Dockerfile,
# because that Docker image may get run on a different computer.
# It's totally fine to have it set to "yes" when compiling tools,
# though, because that does not get baked into the Docker image.
ifeq ($(NATIVE),yes)
CXXFLAGS+= -march=native
endif

ifeq ($(MULTITHREADED),yes)
CXXFLAGS+= -fopenmp
LDFLAGS+= -fopenmp
else
CXXFLAGS+= -Wno-unknown-pragmas
endif

all: $(addprefix bin/, $(EXECUTABLES))

bin/%: %.cpp $(EXTRA_DEPS)
	@mkdir -p bin
	$(CXX) -o $@ $^ $(LDFLAGS) $(CXXFLAGS)

clean:
	rm -f bin *.o
