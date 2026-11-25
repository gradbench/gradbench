# Enzyme over the shared Fortran primals in cpp/gradbench/evals/fortran,
# using flang with the FlangEnzyme pass plugin. The C++ drivers (JSON I/O and
# timing) are compiled with $(CXX) and the whole thing is linked with flang.
#
# FLANG_PREFIX: an LLVM install with flang. ENZYME_PREFIX: an Enzyme install
# built against it with -DENZYME_FLANG=ON -DENZYME_FORTRAN=ON.
FLANG_PREFIX?=/opt/llvm
ENZYME_PREFIX?=/opt/enzyme
LLVM_VERSION?=24
FLANG=$(FLANG_PREFIX)/bin/flang
FFLAGS=-O3 -march=native -fpass-plugin=$(ENZYME_PREFIX)/lib/FlangEnzyme-$(LLVM_VERSION).so -I $(ENZYME_PREFIX)/include/EnzymeFortran

CXX?=c++
CXXFLAGS=-std=c++17 -O3 -Wall -march=native -Wno-unknown-pragmas -I../../cpp
LDFLAGS=-lstdc++ -lm

EXECUTABLES=lse llsq ode det
FSRC=../../cpp/gradbench/evals/fortran

all: $(addprefix bin/, $(EXECUTABLES))

obj/grad_%.o: grad_%.f90 $(FSRC)/%.f90
	@mkdir -p obj
	$(FLANG) $(FFLAGS) -c $< -o $@ -J obj

# Enzyme releases up to at least 0.0.311 cannot read flang's TBAA, and type
# analysis fails on the integer/double index arithmetic in det_of_minor.
obj/grad_det.o: FFLAGS += -mllvm -enzyme-loose-types=1

obj/%.o: %.cpp enzyme_fortran.h
	@mkdir -p obj
	$(CXX) $(CXXFLAGS) -c $< -o $@

bin/%: obj/%.o obj/grad_%.o
	@mkdir -p bin
	$(FLANG) -o $@ $^ $(LDFLAGS)

.PRECIOUS: obj/%.o obj/grad_%.o

clean:
	rm -rf bin obj
