CXX?=c++
CC?=cc
CXXFLAGS=-std=c++17 -O3 -Wall -I../../cpp
LDFLAGS=-lm -ladolc

EXECUTABLES=run_hello run_gmm run_ba run_ht run_lstm run_ode run_llsq run_det

all: $(EXECUTABLES)

run_%: run_%.cpp
	$(CXX) -o $@ $^ $(LDFLAGS) $(CXXFLAGS)

%.o: %.cpp
	$(CXX) $< -c -o $@ $(CXXFLAGS)

clean:
	rm -f $(EXECUTABLES)
