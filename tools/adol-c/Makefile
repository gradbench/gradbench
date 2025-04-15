CXX?=c++
CC?=cc
CXXFLAGS=-std=c++17 -O3 -Wall -I../../cpp
LDFLAGS=-lm -ladolc

EXECUTABLES=hello gmm ba ht lstm ode llsq det

all: $(EXECUTABLES)

%: %.cpp
	$(CXX) -o $@ $^ $(LDFLAGS) $(CXXFLAGS)

%.o: %.cpp
	$(CXX) $< -c -o $@ $(CXXFLAGS)

clean:
	rm -f $(EXECUTABLES)
