EXECUTABLES=hello gmm lstm ba ht kmeans particle saddle ode llsq det lse

ENZYME_LIB?=/gradbench/enzyme-build/Enzyme/
CLANGENZYME=$(ENZYME_LIB)/ClangEnzyme-19.so

CXX=clang++

include ../../cpp/common.mk

# Load Enzyme into clang, so that it differentiates each function before the
# optimizer vectorizes and unrolls it.
CXXFLAGS+=-fno-math-errno -fplugin=$(CLANGENZYME)
