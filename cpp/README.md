# C++ utilities

This directory contains various utilities for implementing evals in C++. It also
contains the "reference implementations" of many primal functions that are used
for validation.

Interesting points of note:

- `Makefile`: downloads an appropriate version of
  [json.hpp](https://github.com/nlohmann/json) - this is the JSON library we use
  for C++.

- `adbench/`: a small amount of code taken from ADBench, used for some evals.

- `gradbench/evals`: contains an `.hpp` file for every eval, with a namespace
  that contains various useful definitions. In particular, for an eval `foo` we
  will have `gradbench/evals/foo.hpp`, usually defining:

  - `foo::Input`: a JSON-deserialisable type corresponding to the eval input.

  - `foo::Output`: a JSON-serialisable type corresponding to the eval output -
    for evals with more than one function, this type will have some prefix (e.g.
    `JacOutput`).

  - `foo::objective` or some similarly named function. This will be a template
    function that can be instantiated using dual numbers or some other
    representation, if useful.

  - `foo::Objective` or some similarly named class, which will be of a form
    suitable to pass to `function_main` from `gradbench/main.hpp`.

## Using `gradbench/main.hpp`

From a C++ perspective, you use `gradbench/main.hpp` by calling the function
`generic_main` where the first two arguments are `argc` and `argv`, and the
third is a mapping from function names to a function that accepts input (as a
string) and then processes that input in whatever way is appropriate (i.e.,
deserialises the input and runs the function). In practice, this function is
always `function_main<T>` where `T` is a class that implements the `Function`
interface. See the code for more details.

### From the command line

You can directly run the implementations that use this interface, without going
through `gradbench`. If an implementation `foo.cpp` compiles to a program `foo`,
you can run it as follows:

```
$ ./foo input.json function
```

Here _function_ is the name of the evaluation function you wish to run (e.g.
`jacobian` or `objective`), and _input.json_ is a JSON file that contains the
equivalent of the `input` field of an `evaluate` message. This can be useful for
running C++ implementations or a debugger or a profiler. You must usually obtain
`input.json` from the log file produced by `gradbench`, as most evals are not
able to produce such files directly.

## Multithreading

The primal functions have been multithreaded with OpenMP when applicable. We try
to write the code such that it will run well when compiled for both sequential
and multithreaded execution, ideally without implementing each function twice.

OpenMP is fairly well suited for this purpose, as pragmas are (almost) ignored
when `-fopenmp` is not passed to the C compiler, although we have encountered
some wrinkles.

### Reductions

One problem is that Enzyme does not support the OpenMP reduction clause. This
forces us to write all reductions manually, with per-thread partial sums. One of
the tools we use for this is [multithread.hpp](gradbench/multithread.hpp), which
defines a very small wrapper around some OpenMP functions. When OpenMP is
disabled, these functions have dummy definitions suitable for sequential
execution, and which the C++ compiler optimizer will be able to propagate to
remove the overhead of the parallel reductions.

### Specialisations

Another problem is that even when OpenMP is disabled, the C++ compiler may still
be unhappy about the pragmas if they involve types it does not understand - and
this happens for template functions that are instantiated with the nonstandard
number types used for tracing by the operator overloading tools. For this
reason, some of our template functions have specialisations for `double` (or
whichever primal type is relevant) in which the OpenMP pragmas are used, while
the unspecialized functions have no pragmas. This means that such tools will not
take advantage of multithreaded execution during tracing.
