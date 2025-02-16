CXX?=c++
CC?=cc
CFLAGS=-std=c++17 -O3 -Wall -I../../cpp
LDFLAGS=-lm -ladolc

UTIL_OBJECTS=../../cpp/adbench/shared/utils.o ../../cpp/adbench/io.o
HT_OBJECTS=AdolCHT.o run_ht.o
EXECUTABLES=run_hello run_gmm run_ba run_ht run_lstm

all: $(EXECUTABLES)

run_ht: $(HT_OBJECTS) $(UTIL_OBJECTS)
	$(CXX) -o $@ $^ $(LDFLAGS)

run_%: run_%.cpp
	$(CXX) -o $@ $^ $(LDFLAGS) $(CFLAGS)

%.o: %.cpp
	$(CXX) $< -c -o $@ $(CFLAGS)

clean:
	rm -f $(UTIL_OBJECTS) $(HT_OBJECTS) $(EXECUTABLES)
