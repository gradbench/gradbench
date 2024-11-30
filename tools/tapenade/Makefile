CXX?=c++
CC?=cc
CFLAGS=-O3 -Wall
LDFLAGS=-lm

UTIL_OBJECTS=utils/adBuffer.o utils/adStack.o ../../cpp/adbench/shared/utils.o ../../cpp/adbench/io.o
GMM_OBJECTS=TapenadeGMM.o gmm/gmm.o gmm/gmm_b.o run_gmm.o
BA_OBJECTS=TapenadeBA.o ba/ba.o ba/ba_b.o run_ba.o
LSTM_OBJECTS=TapenadeLSTM.o lstm/lstm.o lstm/lstm_b.o run_lstm.o
HT_OBJECTS=TapenadeHT.o hand/hand.o hand/hand_b.o run_hand.o
EXECUTABLES=run_gmm run_ba run_lstm

all: $(EXECUTABLES)

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
