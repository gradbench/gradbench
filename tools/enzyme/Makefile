EXECUTABLES=hello gmm lstm ba ht kmeans particle saddle ode llsq det lse

ENZYME_LIB?=/gradbench/enzyme-build/Enzyme/
LLDENZYME=$(ENZYME_LIB)/LLDEnzyme-19.so

CXX=clang++
LLD=lld
LDFLAGS+=-fuse-ld=$(LLD) -O3 -fno-math-errno -flto -Wl,--load-pass-plugin=$(LLDENZYME)

include ../../cpp/common.mk
