# ADOL-C

[ADOL-C](https://github.com/coin-or/ADOL-C) is a C++ library for AD
based on operator overloading. Its usage is discussed in [Algorithm
755: ADOL-C: a package for the automatic differentiation of algorithms
written in C/C++](https://dl.acm.org/doi/10.1145/229473.229474).

When using ADOL-C, you first evaluate the objective function to
construct the *tape*, which will contain a record of all arithmetic
operations. Then you interpret the tape in various ways to compute
gradients. In the [ADBench
implementations](https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/tools/ADOLC/main.cpp),
the construction of this tape is done as part of preparation, and not
measured as part of the runtime. This approach has been kept in this
tool implementation.
