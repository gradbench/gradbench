CXX?=c++
CC?=cc
CFLAGS=-std=c++17 -O3 -Wall
LDFLAGS=-lm

UTIL_OBJECTS=../../cpp/adbench/shared/utils.o ../../cpp/adbench/io.o
HELLO_OBJECTS=ManualHello.o run_hello.o hello_d.o
GMM_OBJECTS=ManualGMM.o run_gmm.o gmm_d.o
BA_OBJECTS=ManualBA.o run_ba.o ba_d.o
LSTM_OBJECTS=ManualLSTM.o run_lstm.o lstm_d.o
HT_OBJECTS=ManualHT.o run_ht.o ht_d.o
KMEANS_OBJECTS=run_kmeans.o kmeans_d.o
EXECUTABLES=run_hello run_gmm run_ba run_lstm run_ht run_kmeans

all: $(EXECUTABLES)

run_hello: $(HELLO_OBJECTS) $(UTIL_OBJECTS)
	$(CXX) -o $@ $^ $(LDFLAGS)

run_ba: $(BA_OBJECTS) $(UTIL_OBJECTS)
	$(CXX) -o $@ $^ $(LDFLAGS)

run_gmm: $(GMM_OBJECTS) $(UTIL_OBJECTS)
	$(CXX) -o $@ $^ $(LDFLAGS)

run_lstm: $(LSTM_OBJECTS) $(UTIL_OBJECTS)
	$(CXX) -o $@ $^ $(LDFLAGS)

run_ht: $(HT_OBJECTS) $(UTIL_OBJECTS)
	$(CXX) -o $@ $^ $(LDFLAGS)

run_kmeans: $(KMEANS_OBJECTS) $(UTIL_OBJECTS)
	$(CXX) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $< -c -o $@ $(CFLAGS) -I../../cpp

clean:
	rm -f $(UTIL_OBJECTS) $(GMM_OBJECTS) $(BA_OBJECTS) $(LSTM_OBJECTS) $(HT_OBJECTS) $(KMEANS_OBJECTS) $(EXECUTABLES)
