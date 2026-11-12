# XAD

[XAD][] is an open-source C++ operator-overloading library for automatic
differentiation, supporting forward and adjoint (reverse) modes.

The image builds XAD v2.2.0 as a static Release library, with the tape options
that suit these evals (see `Dockerfile`). Gradients use adjoint mode; `ht`
computes its Jacobian in vector forward mode and `ba` in vector adjoint mode,
two rows per sweep. The tool is single-threaded (one global tape) and does not
support `--multithreaded`.

Based on work by [Mikhail Pilipov][fork].

[XAD]: https://auto-differentiation.github.io
[fork]: https://github.com/mpilipov/gradbench
