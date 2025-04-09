COMMIT=f8a524c5e2dd373933115a3c2c0c7e6fe30f0690
CXX?=c++
CC?=cc
CFLAGS=-std=c++17 -O3 -Wall -I../../cpp
LDFLAGS=-lm

EXECUTABLES=hello llsq

all: $(EXECUTABLES)

%: %.cpp ad.hpp
	$(CXX) -o $@ $< $(LDFLAGS) $(CFLAGS)

clean:
	rm -f $(EXECUTABLES)

ad.hpp:
	wget https://gitlab.stce.rwth-aachen.de/stce/ad/-/raw/$(COMMIT)/ad.hpp -O $@
