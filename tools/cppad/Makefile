CXX?=c++
CC?=cc
CFLAGS=-std=c++17 -O3 -Wall $(shell pkg-config --cflags cppad)
LDFLAGS=-lm $(shell pkg-config --libs cppad)

UTIL_OBJECTS=../../cpp/adbench/shared/utils.o ../../cpp/adbench/io.o
HELLO_OBJECTS=CppADHello.o run_hello.o
GMM_OBJECTS=CppADGMM.o run_gmm.o
LSTM_OBJECTS=CppADLSTM.o run_lstm.o
EXECUTABLES=run_hello run_gmm run_lstm

all: $(EXECUTABLES)

run_hello: $(HELLO_OBJECTS) $(UTIL_OBJECTS)
	$(CXX) -o $@ $^ $(LDFLAGS)

run_gmm: $(GMM_OBJECTS) $(UTIL_OBJECTS)
	$(CXX) -o $@ $^ $(LDFLAGS)

run_lstm: $(LSTM_OBJECTS) $(UTIL_OBJECTS)
	$(CXX) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $< -c -o $@ $(CFLAGS) -I../../cpp

clean:
	rm -f $(UTIL_OBJECTS) $(GMM_OBJECTS) $(BA_OBJECTS) $(LSTM_OBJECTS) $(HT_OBJECTS) $(EXECUTABLES)
