# Finite Differences

This tool contains programs that are differentiated by Finite Differences. This is likely to be both slower and less efficient than any AD tool, but it is certainly the most convenient option.

To run this outside Docker, you'll first need to run the following commands from the GradBench repository root to setup the C++ build and build the programs:

```sh
make -C cpp
make -C tools/finite
```

Then, to run the tool itself:

```sh
python3 python/gradbench/gradbench/cpp.py finite
```
