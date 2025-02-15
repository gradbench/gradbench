CXX=clang++
LLD=lld
CXXFLAGS=-std=c++17 -O3 -fno-math-errno -Wall -flto
LDFLAGS=-fuse-ld=$(LLD) -O3 -fno-math-errno -flto -Wl,--load-pass-plugin=$(LLDENZYME) -lm

ENZYME_LIB?=/home/gradbench/enzyme-build/Enzyme/
LLDENZYME=$(ENZYME_LIB)/LLDEnzyme-19.so

UTIL_OBJECTS=../../cpp/adbench/shared/utils.o ../../cpp/adbench/io.o
HELLO_OBJECTS=EnzymeHello.o run_hello.o
GMM_OBJECTS=EnzymeGMM.o run_gmm.o
LSTM_OBJECTS=EnzymeLSTM.o run_lstm.o
BA_OBJECTS=run_ba.o
HT_OBJECTS=EnzymeHT.o run_ht.o
KMEANS_OBJECTS=run_kmeans.o
EXECUTABLES=run_hello run_gmm run_lstm run_ba run_ht run_kmeans

all: $(EXECUTABLES)

run_hello: $(HELLO_OBJECTS) $(UTIL_OBJECTS)
	$(CXX) -o $@ $^ $(LDFLAGS)

run_gmm: $(GMM_OBJECTS) $(UTIL_OBJECTS)
	$(CXX) -o $@ $^ $(LDFLAGS)

run_lstm: $(LSTM_OBJECTS) $(UTIL_OBJECTS)
	$(CXX) -o $@ $^ $(LDFLAGS)

run_ba: $(BA_OBJECTS)
	$(CXX) -o $@ $^ $(LDFLAGS)

run_ht: $(HT_OBJECTS) $(UTIL_OBJECTS)
	$(CXX) -o $@ $^ $(LDFLAGS)

run_kmeans: $(KMEANS_OBJECTS)
	$(CXX) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $< -c -o $@ $(CXXFLAGS) -I../../cpp

clean:
	rm -f $(UTIL_OBJECTS) $(HELLO_OBJECTS) $(GMM_OBJECTS) $(LSTM_OBJECTS) $(HT_OBJECTS) $(KMEANS_OBJECTS) $(EXECUTABLES)
