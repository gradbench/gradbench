CXX?=c++
CC?=cc
CXXFLAGS=-std=c++17 -O3 -Wall $(shell pkg-config --cflags cppad) -I../../cpp
LDFLAGS=-lm $(shell pkg-config --libs cppad)

EXECUTABLES=run_hello run_gmm run_lstm run_an_ode run_llsq_obj run_det_by_minor

all: $(EXECUTABLES)

run_%: run_%.cpp
	$(CXX) -o $@ $^ $(LDFLAGS) $(CXXFLAGS)

clean:
	rm -f $(UTIL_OBJECTS) $(EXECUTABLES)
