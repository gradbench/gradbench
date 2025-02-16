CXX=clang++
LLD=lld
CXXFLAGS=-std=c++17 -O3 -fno-math-errno -Wall -flto -I../../cpp
LDFLAGS=-fuse-ld=$(LLD) -O3 -fno-math-errno -flto -Wl,--load-pass-plugin=$(LLDENZYME) -lm

ENZYME_LIB?=/home/gradbench/enzyme-build/Enzyme/
LLDENZYME=$(ENZYME_LIB)/LLDEnzyme-19.so

UTIL_OBJECTS=../../cpp/adbench/shared/utils.o ../../cpp/adbench/io.o
LSTM_OBJECTS=EnzymeLSTM.o run_lstm.o
HT_OBJECTS=EnzymeHT.o run_ht.o
EXECUTABLES=run_hello run_gmm run_lstm run_ba run_ht run_kmeans

all: $(EXECUTABLES)

run_lstm: $(LSTM_OBJECTS) $(UTIL_OBJECTS)
	$(CXX) -o $@ $^ $(LDFLAGS)

run_ht: $(HT_OBJECTS) $(UTIL_OBJECTS)
	$(CXX) -o $@ $^ $(LDFLAGS)

run_%: run_%.cpp
	$(CXX) -o $@ $^ $(LDFLAGS) $(CXXFLAGS)

%.o: %.cpp
	$(CXX) $< -c -o $@ $(CXXFLAGS) -I../../cpp

clean:
	rm -f $(UTIL_OBJECTS) $(LSTM_OBJECTS) $(HT_OBJECTS) $(EXECUTABLES)
