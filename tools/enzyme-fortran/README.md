# Enzyme (Fortran)

This tool runs [Enzyme][] on Fortran code, through [Flang][] and the
`FlangEnzyme` pass plugin. The primal programs live in
[`cpp/gradbench/evals/fortran`](../../cpp/gradbench/evals/fortran), so that
other Fortran AD tools can differentiate the same sources.

Each `grad_<eval>.f90` `include`s the primal, so that the primal body is in the
same translation unit as the `enzyme_autodiff` call, and exports a `bind(c)`
gradient routine. The `<eval>.cpp` drivers handle JSON I/O and timing, and use
the shared C++ primal for the `primal` function, as the other C++ tools do.

Supported evals: `det`, `llsq`, `lse`, `ode`.

## Running outside Docker

You need an LLVM install with `flang` and an Enzyme install built against it
with `-DENZYME_FLANG=ON -DENZYME_FORTRAN=ON` (for `FlangEnzyme-<version>.so`
and the `enzyme` Fortran module). Then, from the repository root:

```sh
make -C cpp
export FLANG_PREFIX=/path/to/llvm ENZYME_PREFIX=/path/to/enzyme LLVM_VERSION=24
python3 python/gradbench/gradbench/cpp.py enzyme-fortran
```

The Docker image uses Debian's `flang-22` and builds an Enzyme release
against it.

[enzyme]: https://enzyme.mit.edu/
[flang]: https://flang.llvm.org/
