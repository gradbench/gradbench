CXX?=c++
CC?=cc
CFLAGS=-std=c++17 -O3 -Wall -I../../cpp
LDFLAGS=-lm

EXECUTABLES=run_hello run_gmm run_ba run_lstm run_ht run_kmeans run_an_ode run_llsq_obj run_det_by_minor

all: $(EXECUTABLES)

run_%: run_%.cpp
	$(CXX) -o $@ $^ $(LDFLAGS) $(CFLAGS)

clean:
	rm -f $(EXECUTABLES)
