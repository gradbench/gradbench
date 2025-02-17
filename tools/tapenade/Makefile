CXX?=c++
CC?=cc
CFLAGS=-std=c11 -O3 -Wall -I../../cpp
CXXFLAGS=-std=c++17 -O3 -Wall -I../../cpp
LDFLAGS=-lm

UTIL_OBJECTS=utils/adBuffer.o utils/adStack.o

GMM_OBJECTS=gmm/gmm.o gmm/gmm_b.o run_gmm.o
BA_OBJECTS=ba/ba.o ba/ba_b.o run_ba.o
LSTM_OBJECTS=lstm/lstm.o lstm/lstm_b.o run_lstm.o
HT_OBJECTS=hand/hand.o hand/hand_d.o run_ht.o

EXECUTABLES=run_gmm run_ba run_lstm run_ht

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
	$(CC) $< -c -o $@ $(CFLAGS)

%.o: %.cpp
	$(CXX) $< -c -o $@ $(CXXFLAGS)

clean:
	rm -f $(UTIL_OBJECTS) $(GMM_OBJECTS) $(BA_OBJECTS) $(LSTM_OBJECTS) $(HT_OBJECTS) $(EXECUTABLES)
