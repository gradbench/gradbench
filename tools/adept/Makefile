CXX?=c++
CC?=cc
CFLAGS=-std=c++17 -O2 -Wall -fopenmp
LDFLAGS=-lm -ladept -fopenmp

UTIL_OBJECTS=../../cpp/adbench/shared/utils.o ../../cpp/adbench/io.o
HELLO_OBJECTS=AdeptHello.o run_hello.o
GMM_OBJECTS=AdeptGMM.o run_gmm.o
HT_OBJECTS=AdeptHT.o run_ht.o
BA_OBJECTS=AdeptBA.o run_ba.o
LSTM_OBJECTS=AdeptLSTM.o run_lstm.o

EXECUTABLES=run_hello run_gmm run_ht run_ba run_lstm

all: $(EXECUTABLES)

run_hello: $(HELLO_OBJECTS) $(UTIL_OBJECTS)
	$(CXX) -o $@ $^ $(LDFLAGS)

run_gmm: $(GMM_OBJECTS) $(UTIL_OBJECTS)
	$(CXX) -o $@ $^ $(LDFLAGS)

run_ht: $(HT_OBJECTS) $(UTIL_OBJECTS)
	$(CXX) -o $@ $^ $(LDFLAGS)

run_ba: $(BA_OBJECTS) $(UTIL_OBJECTS)
	$(CXX) -o $@ $^ $(LDFLAGS)

run_lstm: $(LSTM_OBJECTS) $(UTIL_OBJECTS)
	$(CXX) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $< -c -o $@ $(CFLAGS) -I../../cpp

clean:
	rm -f $(UTIL_OBJECTS) $(GMM_OBJECTS) $(BA_OBJECTS) $(LSTM_OBJECTS) $(HT_OBJECTS) $(EXECUTABLES)
