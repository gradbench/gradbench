# Manual

[Enzyme][] is a tool for computing derivatives of arbitrary LLVM IR
code. The implementations here use the sequential objective functions
implemented in C++ and differentiated them with Enzyme, through an
LLVM plugin.

Enzyme needs to be compiled against a specific version of LLVM. You
are strongly adviced to use the Dockerfile for this one.

[Enzyme]: https://enzyme.mit.edu/
