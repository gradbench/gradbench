CXX?=c++
CC?=cc
CFLAGS=-std=c++17 -O3 -Wall -I../../cpp
LDFLAGS=-lm

EXECUTABLES=run_hello run_gmm run_lstm run_ba run_ht

all: $(EXECUTABLES)

run_%: run_%.cpp
	$(CXX) -o $@ $^ $(LDFLAGS) $(CFLAGS)

clean:
	rm -f $(EXECUTABLES)
