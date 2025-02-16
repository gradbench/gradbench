CXX?=c++
CC?=cc
CFLAGS=-std=c++17 -O3 -Wall $(shell pkg-config --cflags cppad) -I../../cpp
LDFLAGS=-lm $(shell pkg-config --libs cppad)

UTIL_OBJECTS=../../cpp/adbench/shared/utils.o ../../cpp/adbench/io.o
LSTM_OBJECTS=CppADLSTM.o run_lstm.o
EXECUTABLES=run_hello run_gmm run_lstm

all: $(EXECUTABLES)

run_lstm: $(LSTM_OBJECTS) $(UTIL_OBJECTS)
	$(CXX) -o $@ $^ $(LDFLAGS)

run_%: run_%.cpp
	$(CXX) -o $@ $^ $(LDFLAGS) $(CFLAGS)

%.o: %.cpp
	$(CXX) $< -c -o $@ $(CFLAGS) -I../../cpp

clean:
	rm -f $(UTIL_OBJECTS) $(BA_OBJECTS) $(LSTM_OBJECTS) $(HT_OBJECTS) $(EXECUTABLES)
