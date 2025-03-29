# Manual

This tool contains programs that have been differentiated by hand.

To run this outside Docker, you'll first need to run the following commands from the GradBench repository root to setup the C++ build and build the programs:

```sh
make -C cpp
make -C tools/manual
```

Then, to run the tool itself:

```sh
python3 python/gradbench/gradbench/cpp.py manual
```

## Commentary

We expect that in most cases, the hand-differentiated versions should
be the fastest, as they may exploit mathematical properties that it is
not reasonable to expect of an AD tool. However, these are only
algorithmic improvements: a tool may beat `manual` through operational
advantages, such as efficient implementations of primitives like
matrix multiplication, parallel execution, etc. that the `manual`
implementation of a tool does not do. It is not a goal that the
`manual` tool is at all costs the *fastest it can possibly be*.

It is not expected that we will be able to implement
hand-differentiated versions of all evals. AD is after all most useful
for those cases where hand-differentiation is impractical.
