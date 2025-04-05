CXX?=c++
CC?=cc
CFLAGS=-std=c11 -O3 -Wall -I../../cpp
CXXFLAGS=-std=c++17 -O3 -Wall -I../../cpp
LDFLAGS=-lm

UTIL_OBJECTS=utils/adBuffer.o utils/adStack.o

GMM_OBJECTS=evals/gmm/gmm.o evals/gmm/gmm_b.o gmm.o
BA_OBJECTS=evals/ba/ba.o evals/ba/ba_b.o ba.o
LSTM_OBJECTS=evals/lstm/lstm.o evals/lstm/lstm_b.o lstm.o
HT_OBJECTS=evals/hand/hand.o evals/hand/hand_d.o ht.o

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
