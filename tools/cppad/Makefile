CXX?=c++
CC?=cc
CXXFLAGS=-std=c++17 -O3 -Wall $(shell pkg-config --cflags cppad) -I../../cpp
LDFLAGS=-lm $(shell pkg-config --libs cppad)

EXECUTABLES=hello gmm lstm kmeans ode llsq det

all: $(EXECUTABLES)

%: %.cpp
	$(CXX) -o $@ $^ $(LDFLAGS) $(CXXFLAGS)

clean:
	rm -f $(UTIL_OBJECTS) $(EXECUTABLES)
