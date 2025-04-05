CXX?=c++
CC?=cc
CFLAGS=-std=c++17 -O3 -Wall -I../../cpp
LDFLAGS=-lm

EXECUTABLES=hello gmm lstm ba ht kmeans

all: $(EXECUTABLES)

%: %.cpp
	$(CXX) -o $@ $^ $(LDFLAGS) $(CFLAGS)

clean:
	rm -f $(EXECUTABLES)
