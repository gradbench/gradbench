# Zygote

[Zygote][] is a differentiable programming library for the [Julia][] programming
language.

## Running outside of Docker

[See the general instructions.](/julia/#running-outside-of-docker)

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

[julia]: https://julialang.org/
[zygote]: https://fluxml.ai/Zygote.jl/
