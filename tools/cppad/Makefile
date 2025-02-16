CXX?=c++
CC?=cc
CFLAGS=-std=c++17 -O3 -Wall $(shell pkg-config --cflags cppad) -I../../cpp
LDFLAGS=-lm $(shell pkg-config --libs cppad)

UTIL_OBJECTS=../../cpp/adbench/shared/utils.o ../../cpp/adbench/io.o
EXECUTABLES=run_hello run_gmm run_lstm

all: $(EXECUTABLES)

run_%: run_%.cpp
	$(CXX) -o $@ $^ $(LDFLAGS) $(CFLAGS)

%.o: %.cpp
	$(CXX) $< -c -o $@ $(CFLAGS) -I../../cpp

clean:
	rm -f $(UTIL_OBJECTS) $(EXECUTABLES)
