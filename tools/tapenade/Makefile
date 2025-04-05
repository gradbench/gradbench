CXX?=c++
CC?=cc
CFLAGS=-std=c11 -O3 -Wall -I../../cpp
CXXFLAGS=-std=c++17 -O3 -Wall -I../../cpp
LDFLAGS=-lm

UTIL_OBJECTS=utils/adBuffer.o utils/adStack.o

GMM_OBJECTS=gmm/gmm.o gmm/gmm_b.o gmm.o
BA_OBJECTS=ba/ba.o ba/ba_b.o ba.o
LSTM_OBJECTS=lstm/lstm.o lstm/lstm_b.o lstm.o
HT_OBJECTS=hand/hand.o hand/hand_d.o ht.o

EXECUTABLES=gmm ba lstm ht

all: $(EXECUTABLES)

ba: $(BA_OBJECTS) $(UTIL_OBJECTS)
	$(CXX) -o $@ $^ $(LDFLAGS)

gmm: $(GMM_OBJECTS) $(UTIL_OBJECTS)
	$(CXX) -o $@ $^ $(LDFLAGS)

lstm: $(LSTM_OBJECTS) $(UTIL_OBJECTS)
	$(CXX) -o $@ $^ $(LDFLAGS)

ht: $(HT_OBJECTS) $(UTIL_OBJECTS)
	$(CXX) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $< -c -o $@ $(CFLAGS)

%.o: %.cpp
	$(CXX) $< -c -o $@ $(CXXFLAGS)

clean:
	rm -f $(UTIL_OBJECTS) $(GMM_OBJECTS) $(BA_OBJECTS) $(LSTM_OBJECTS) $(HT_OBJECTS) $(EXECUTABLES)
