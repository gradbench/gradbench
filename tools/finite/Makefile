CXX?=c++
CC?=cc
CFLAGS=-std=c++17 -O3 -Wall -I../../cpp
LDFLAGS=-lm

UTIL_OBJECTS=../../cpp/adbench/shared/utils.o ../../cpp/adbench/io.o
LSTM_OBJECTS=FiniteLSTM.o run_lstm.o
HT_OBJECTS=FiniteHT.o run_ht.o
EXECUTABLES=run_hello run_gmm run_ba run_lstm run_ht

all: $(EXECUTABLES)

run_lstm: $(LSTM_OBJECTS) $(UTIL_OBJECTS)
	$(CXX) -o $@ $^ $(LDFLAGS)

run_ht: $(HT_OBJECTS) $(UTIL_OBJECTS)
	$(CXX) -o $@ $^ $(LDFLAGS)

run_%: run_%.cpp
	$(CXX) -o $@ $^ $(LDFLAGS) $(CFLAGS)

%.o: %.cpp
	$(CXX) $< -c -o $@ $(CFLAGS)

clean:
	rm -f $(UTIL_OBJECTS) $(LSTM_OBJECTS) $(HT_OBJECTS) $(EXECUTABLES)
