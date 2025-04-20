GIT_TAG=v1.7
CXX?=c++
CC?=cc
CFLAGS=-std=c++17 -O3 -Wall -I../../cpp -Iinclude
LDFLAGS=-lm

EXECUTABLES=hello ba det gmm llsq lse lstm ode particle

all: $(EXECUTABLES)

%: %.cpp include/ad.hpp
	$(CXX) -o $@ $< $(LDFLAGS) $(CFLAGS)

clean:
	rm -f $(EXECUTABLES)

include/ad.hpp:
	mkdir -p include
	wget https://gitlab.stce.rwth-aachen.de/stce/ad/-/raw/$(GIT_TAG)/$@ -O $@

include/extensions/ad_interoperability_eigen.hpp:
	mkdir -p include/extensions
	wget https://gitlab.stce.rwth-aachen.de/stce/ad/-/raw/$(GIT_TAG)/$@ -O $@

