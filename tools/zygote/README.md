# Zygote

[Zygote][] is a differentiable programming library for the [Julia][] programming language.

## Running outside of Docker

Make sure the `julia` executable is on your `PATH`. Then run the
following command to download and install all packages:

```
$ julia --project=tools/zygote -e 'import Pkg; Pkg.instantiate()'
```

Then you can run the tool as follows:

```
$ julia --project=tools/zygote tools/zygote/run.jl
```

[julia]: https://julialang.org/
[zygote]: https://fluxml.ai/Zygote.jl/
