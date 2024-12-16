# The ADEPT_DIR default is set such that you can download adept into
# this directory for easy testing, but is overridden by an environment
# variable in the Dockerfile.
ADEPT_DIR?=adept-2.1.1
ADEPT_INCLUDE_DIR=$(ADEPT_DIR)/include
ADEPT_LIB_DIR=$(ADEPT_DIR)/adept/.libs/

CXX?=c++
CC?=cc
CFLAGS=-std=c++17 -O2 -Wall -I$(ADEPT_INCLUDE_DIR) -fopenmp
LDFLAGS=-lm $(ADEPT_LIB_DIR)/libadept.a -fopenmp

UTIL_OBJECTS=../../cpp/adbench/shared/utils.o ../../cpp/adbench/io.o
HELLO_OBJECTS=AdeptHello.o run_hello.o
GMM_OBJECTS=AdeptGMM.o run_gmm.o
HT_OBJECTS=AdeptHT.o run_ht.o

EXECUTABLES=run_hello run_gmm run_ht

all: $(EXECUTABLES)

run_hello: $(HELLO_OBJECTS) $(UTIL_OBJECTS)
	$(CXX) -o $@ $^ $(LDFLAGS)

run_gmm: $(GMM_OBJECTS) $(UTIL_OBJECTS)
	$(CXX) -o $@ $^ $(LDFLAGS)

run_ht: $(HT_OBJECTS) $(UTIL_OBJECTS)
	$(CXX) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $< -c -o $@ $(CFLAGS) -I../../cpp

clean:
	rm -f $(UTIL_OBJECTS) $(GMM_OBJECTS) $(BA_OBJECTS) $(LSTM_OBJECTS) $(HT_OBJECTS) $(EXECUTABLES)
