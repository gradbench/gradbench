# Manual

This tool contains programs that have been differentiated by hand.

To run this outside Docker, you'll first need to run the following commands from the GradBench repository root to setup the C++ build and build the programs:

```sh
make -C cpp
make -C tools/manual
```

Then, to run the tool itself:

```sh
python3 tools/manual/run.py
```
