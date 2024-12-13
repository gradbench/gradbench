# The ADEPT_DIR default is set such that you can download adept into
# this directory for easy testing, but is overridden by an environment
# variable in the Dockerfile.
ADEPT_DIR?=adept-2.1.1
ADEPT_INCLUDE_DIR=$(ADEPT_DIR)/include
ADEPT_LIB_DIR=$(ADEPT_DIR)/adept/.libs/

CXX?=c++
CC?=cc
CFLAGS=-std=c++17 -O3 -Wall -I$(ADEPT_INCLUDE_DIR) -fopenmp
LDFLAGS=-lm $(ADEPT_LIB_DIR)/libadept.a -fopenmp

UTIL_OBJECTS=../../cpp/adbench/shared/utils.o ../../cpp/adbench/io.o
HELLO_OBJECTS=AdeptHello.o run_hello.o
GMM_OBJECTS=AdeptGMM.o run_gmm.o

EXECUTABLES=run_hello run_gmm

all: $(EXECUTABLES)

run_hello: $(HELLO_OBJECTS) $(UTIL_OBJECTS)
	$(CXX) -o $@ $^ $(LDFLAGS)

run_gmm: $(GMM_OBJECTS) $(UTIL_OBJECTS)
	$(CXX) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $< -c -o $@ $(CFLAGS) -I../../cpp

%.o: %.cpp
	$(CXX) $< -c -o $@ $(CFLAGS) -I../../cpp

clean:
	rm -f $(UTIL_OBJECTS) $(GMM_OBJECTS) $(BA_OBJECTS) $(LSTM_OBJECTS) $(HT_OBJECTS) $(EXECUTABLES)
