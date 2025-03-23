CXX?=c++
CC?=cc
CXXFLAGS=-std=c++17 -O2 -Wall -fopenmp -I../../cpp
LDFLAGS=-lm -ladept -fopenmp

EXECUTABLES=run_hello run_gmm run_ht run_ba run_lstm run_an_ode run_llsq_obj run_det_by_minor

all: $(EXECUTABLES)

run_%: run_%.cpp
	$(CXX) -o $@ $^ $(LDFLAGS) $(CXXFLAGS)

clean:
	rm -f $(EXECUTABLES)
