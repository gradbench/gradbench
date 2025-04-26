# Enzyme

[Enzyme][] is a tool for computing derivatives of arbitrary LLVM IR code. The implementations here use the sequential objective functions implemented in C++ and differentiated with Enzyme, through an LLVM plugin.

Enzyme needs to be compiled against a specific version of LLVM. You are strongly advised to use the Dockerfile for this one.

[Enzyme]: https://enzyme.mit.edu/

## Commentary

### Multithreading

The approach here is very similar to the one for [manual](../manual),
except that the derivative functions are (obviously) produced with
Enzyme. In particular, the primal functions use the [C++ reference
implementations][../../cpp/gradbench/evals] - see the Commentary for
each individual eval to see how well they have been multithreaded. In
general, Enzyme is applied directly to these primal functions.

However, for some evals, particularly those involving Jacobians, it is
more efficient to parallelise only the independent evaluations of the
derived function, rather than inside the derived function itself.

Specific notes:

- `ht::jacobian`: multithreaded the multiple evaluations of the
  derivative. (Currently crashes Enzyme.)
