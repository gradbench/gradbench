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

# This needs to be set to "no" anytime the compiled binary may run on a
# different computer than the one that compiled it -- e.g. when it gets baked
# into a portable OCI image -- since then we can't rely on the current native
# architecture. For example, when compiling the "manual" tool as the golden
# impl for an eval, we can't use anything native. But it's totally fine to
# have it set to "yes" when compiling a tool right before running it locally,
# which does not get baked into an image.
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
