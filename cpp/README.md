# C++ utilities

This directory contains various utilities for implementing evals in C++.
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
