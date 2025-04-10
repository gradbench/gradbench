# Common Makefile definitions that are used by many of the C++-based
# tools. You do not have to use it, but the cpp.py tooling expects an
# interface that behaves like this.
#
# The EXECUTABLES variable (and the 'all' target) is not needed for
# cpp.py, but only for local testing.

CXX?=c++
CC?=cc
CFLAGS?=-std=c++17 -O3 -Wall -I../../cpp
LDFLAGS?=-lm

MULTITHREADED=no

ifeq ($(MULTITHREADED),yes)
CFLAGS+= -fopenmp -DUSE_OPENMP
LDFLAGS+= -fopenmp
else
CFLAGS+= -Wno-unknown-pragmas
endif

all: $(EXECUTABLES)

%: %.cpp
	$(CXX) -o $@ $^ $(LDFLAGS) $(CFLAGS)

clean:
	rm -f $(EXECUTABLES)

