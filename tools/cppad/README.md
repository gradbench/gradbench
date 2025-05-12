# CppAD

[CppAD](https://github.com/coin-or/CppAD) is a C++ library for AD based on
operator overloading.

To run this outside Docker, you'll first need to run the following command from
the GradBench repository root to setup the C++ build and build the programs:

```sh
make -C cpp
```

You will then need to have CppAD installed in a way that `pkg-config` can find
it.
