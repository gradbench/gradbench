# Zygote

[Zygote][] is a differentiable programming library for the [Julia][] programming
language.

## Running outside of Docker

Make sure the `julia` executable is on your `PATH`. Then run the following
command to download and install all packages:

```
$ julia --project=tools/zygote -e 'import Pkg; Pkg.instantiate()'
```

Then you can run the tool as follows:

```
$ julia --project=tools/zygote tools/zygote/run.jl
```

[julia]: https://julialang.org/
[zygote]: https://fluxml.ai/Zygote.jl/

## Commentary

Some of the implementations of the ADBench evals (`gmm`, `ht`, `ba`, `lstm`)
have been improved compared to the original implementations.

- `gmm` has been vectorised. This slightly reduces primal performance, but
  significantly helps Zygote.

- `ht` has been vectorised, and now exploits sparsity when computing the
  Jacobian - this improves performance of the "complicated" variant by orders of
  magnitude. It still suffers slightly by Zygote not supporting the forward mode
  of AD.

- `ba` has been lightly micro-optimised, but but impact is not major.

### Multithreading
