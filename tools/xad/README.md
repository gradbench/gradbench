# XAD

[XAD][] is an open-source C++ operator-overloading library for automatic
differentiation, supporting forward and adjoint (reverse) modes.

The image builds XAD v2.2.0 as a static Release library, with the tape options
that suit these evals (see `Dockerfile`). Gradients use adjoint mode; `ht`
computes its Jacobian in vector forward mode and `ba` in vector adjoint mode,
two rows per sweep. `particle` and `saddle` nest the two modes in all four
combinations (forward or adjoint over forward or adjoint). The tool is
single-threaded (one global tape) and does not support `--multithreaded`.

Based on [Mikhail Pilipov's fork][fork], the first XAD integration for
GradBench.

[XAD]: https://www.xcelerit.com/xad/docs/
[fork]: https://github.com/mpilipov/gradbench
