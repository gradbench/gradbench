# Enzyme

[Enzyme.jl][] is a differentiable programming library for the [Julia][]
programming language.

## Running outside of Docker

Make sure the `julia` executable is on your `PATH`. Then run the following
command to download and install all packages:

```
$ julia -t auto --project=tools/enzyme-jl -e 'import Pkg; Pkg.instantiate()'
```

Then you can run the tool as follows:

```
$ julia -t auto --project=tools/enzyme-jl tools/enzyme-jl/run.jl
```

## Commentary

### Multithreading

As with [plain Enzyme](/tools/enzyme/README.md), Enzyme.jl supports
multithreaded execution when the underlying primal function is multithreaded.

[julia]: https://julialang.org/
[Enzyme.jl]: https://enzyme.mit.edu/
