# C++ utilities

This directory contains various utilities for implementing evals in
C++. Interesting points of ntoe:

- `Makefile`: downloads an appropriate version of
  [json.hpp](https://github.com/nlohmann/json) - this is the JSON
  library we use for C++.

- `adbench/`: a small amount of code taken from ADBench, used for some
  evals.

- `gradbench/evals`: contains an `.hpp` file for every eval, with a
  namespace that contains various useful definitions. In particular,
  for an eval `foo` we will have `gradbench/evals/foo.hpp`, usually
  defining:

  - `foo::Input`: a JSON-deserialisable type corresponding to the eval input.

  - `foo::Output`: a JSON-serialisable type corresponding to the eval
    output - for evals with more than one function, this type will
    have some prefix (e.g. `JacOutput`).

  - `foo::objective` or some similarly named function. This will be a
    template function that can be instantiated using dual numbers or
    some other representation, if useful.

  - `foo::Objective` or some similarly named class, which will be of a
    form suitable to pass to `function_main` from
    `gradbench/main.hpp`.
