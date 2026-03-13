# ADOL-C

[autodiff](https://github.com/autodiff/autodiff/tree/main) is a C++17 library
that uses modern and advanced programming techniques to enable automatic
computation of derivatives. It is based on taping and supports forward and
reverse mode. It is however mainly intended for forward mode, and its reverse
mode implementation is based on expression trees that are traversed with [stack
usage proportional to the size of the
tree](https://github.com/autodiff/autodiff/issues/314). This results in stack
overflows for most of GradBench's larger workloads.
