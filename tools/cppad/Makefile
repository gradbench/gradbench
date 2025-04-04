CXX?=c++
CC?=cc
CXXFLAGS=-std=c++17 -O3 -Wall $(shell pkg-config --cflags cppad) -I../../cpp
LDFLAGS=-lm $(shell pkg-config --libs cppad)

EXECUTABLES=run_hello run_gmm run_lstm run_kmeans run_ode run_llsq run_det

all: $(EXECUTABLES)

run_%: run_%.cpp
	$(CXX) -o $@ $^ $(LDFLAGS) $(CXXFLAGS)

clean:
	rm -f $(UTIL_OBJECTS) $(EXECUTABLES)
