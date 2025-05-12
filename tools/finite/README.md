# Finite Differences

This tool contains programs that are differentiated by Finite Differences. This
is likely to be both slower and less efficient than any AD tool, but it is
certainly the most convenient option.

To run this outside Docker, you'll first need to run the following commands from
the GradBench repository root to setup the C++ build and build the programs:

```sh
make -C cpp
make -C tools/finite
```

Then, to run the tool itself:

```sh
python3 python/gradbench/gradbench/cpp.py finite
```

## Commentary

The implementation uses a general-purpose (but very simple) finite differences
module implemented in the file [finite.hpp][] with support for arbitrary
higher-order differentiation. This is used in a straightforward manner to
differentiate the primal functions from the
[C++ reference implementations](/cpp/gradbench/evals).

### Multihreading

The finite differences module ([finite.hpp][]) supports parallelism over the
number of input parameters, implemented with OpenMP. Further, depending on the
OpenMP configuration, nested parallelism inside the primal function may also be
utilised.

[finite.hpp]: ./finite.hpp
