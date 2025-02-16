CXX?=c++
CC?=cc
CFLAGS=-std=c++17 -O3 -Wall -I../../cpp
LDFLAGS=-lm

UTIL_OBJECTS=../../cpp/adbench/shared/utils.o ../../cpp/adbench/io.o
LSTM_OBJECTS=ManualLSTM.o run_lstm.o lstm_d.o
HT_OBJECTS=ManualHT.o run_ht.o ht_d.o
KMEANS_OBJECTS=run_kmeans.o
EXECUTABLES=run_hello run_gmm run_ba run_lstm run_ht run_kmeans

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
	rm -f $(UTIL_OBJECTS) $(LSTM_OBJECTS) $(HT_OBJECTS) $(KMEANS_OBJECTS) $(EXECUTABLES)
