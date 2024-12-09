CXX=clang++-19
LLD=lld-19
CXXFLAGS=-std=c++17 -O3 -Wall -flto
LDFLAGS=-fuse-ld=$(LLD) -O3 -flto -Wl,--load-pass-plugin=$(LLDENZYME) -lm

LLDENZYME?=/home/gradbench/enzyme-build/Enzyme/LLDEnzyme-19.so

UTIL_OBJECTS=../../cpp/adbench/shared/utils.o ../../cpp/adbench/io.o
HELLO_OBJECTS=EnzymeHello.o run_hello.o
GMM_OBJECTS=EnzymeGMM.o run_gmm.o
BA_OBJECTS=EnzymeBA.o run_ba.o
LSTM_OBJECTS=EnzymeLSTM.o run_lstm.o
HT_OBJECTS=EnzymeHT.o run_ht.o
EXECUTABLES=run_hello run_gmm # run_ba run_lstm run_ht

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

%.o: %.cpp
	$(CXX) $< -c -o $@ $(CXXFLAGS) -I../../cpp

clean:
	rm -f $(UTIL_OBJECTS) $(HELLO_OBJECTS) $(GMM_OBJECTS) $(BA_OBJECTS) $(LSTM_OBJECTS) $(HT_OBJECTS) $(EXECUTABLES)
