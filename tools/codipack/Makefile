CXX?=c++
CC?=cc
CFLAGS=-std=c++17 -O3 -Wall -march=native -DCODI_ForcedInlines=1 -I../../cpp
LDFLAGS=-lm

EXECUTABLES=ba det gmm hello ht kmeans llsq lse lstm ode

all: $(EXECUTABLES)

%: %.cpp
	$(CXX) -o $@ $^ $(LDFLAGS) $(CFLAGS)

clean:
	rm -f $(EXECUTABLES)
