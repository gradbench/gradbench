MULTITHREADED=no

CXX?=c++
CC?=cc
CFLAGS=-std=c++17 -O3 -Wall -I../../cpp
LDFLAGS=-lm

ifeq ($(MULTITHREADED),yes)
CFLAGS+= -fopenmp
LDFLAGS+= -fopenmp
else
CFLAGS+= -Wno-unknown-pragmas
endif

EXECUTABLES=hello gmm ba lstm ht particle saddle ode llsq det

all: $(EXECUTABLES)

%: %.cpp
	$(CXX) -o $@ $^ $(LDFLAGS) $(CFLAGS)

clean:
	rm -f $(EXECUTABLES)
