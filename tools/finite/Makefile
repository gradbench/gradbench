CXX?=c++
CC?=cc
CFLAGS=-std=c++17 -O3 -Wall
LDFLAGS=-lm

UTIL_OBJECTS=../../cpp/adbench/shared/utils.o ../../cpp/adbench/io.o
HELLO_OBJECTS=FiniteHello.o run_hello.o
GMM_OBJECTS=FiniteGMM.o run_gmm.o
BA_OBJECTS=FiniteBA.o run_ba.o
LSTM_OBJECTS=FiniteLSTM.o run_lstm.o
HT_OBJECTS=FiniteHT.o run_ht.o
EXECUTABLES=run_hello run_gmm run_ba run_lstm run_ht

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

%.o: %.c
	$(CC) $< -c -o $@ $(CFLAGS) -I../../cpp

%.o: %.cpp
	$(CXX) $< -c -o $@ $(CFLAGS) -I../../cpp

clean:
	rm -f $(UTIL_OBJECTS) $(GMM_OBJECTS) $(BA_OBJECTS) $(LSTM_OBJECTS) $(HT_OBJECTS) $(EXECUTABLES)
