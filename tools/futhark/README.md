# Futhark

[Futhark][] is a purely functional array language with automatic differentiation built in.

The tool accepts a `--backend` option to change which Futhark compiler backend to use. Unless you also modify the Dockerfile to enable GPU passthrough, only the `c` (the default) and `multicore` backends are likely to work.

To run this outside Docker, you'll first need to install Futhark and run the following command in the same directory as this `README.md` file:

```sh
futhark pkg sync
```

## Commentary

Most of the implementations are clean and perform well, relative to
the expectations of a pure functional language. The implementations
are always (when algorithmically possible) written in a parallel
style, even though the programs are run sequentially by default, and
sometimes the parallel style is slower in that case.

- [det.fut][]: Futhark does not support recursion, so this is
  implemented in a rather different way that is less efficient but
  data parallel, based on computing and applying permutations. The
  amount (and form) of floating-point work is the same, so this is in
  spirit still the same (inefficient) algorithm as specified by the
  eval.

- [saddle.fut][] and [particle.fut][] make use of [solver.fut][],
  which is a rather simple implementation of Gradient Descent. Their
  performance is slightly hampered by representing two-dimensional
  points as two-element arrays, which is [not particularly efficient
  in Futhark][]. The implementation is otherwise very clean, and works
  by defining a generic function that is then instantiated by the four
  different combinations of differential operators.

- [ba.fut][]: Just as much time is spent packing the sparse Jacobian
  in the expected format as in actually computing it. This part is not
  subject to AD.

[futhark]: https://futhark-lang.org/
[det.fut]: det.fut
[saddle.fut]: saddle.fut
[particle.fut]: particle.fut
[solver.fut]: solver.fut
[not particularly efficient in Futhark]: https://futhark-lang.org/blog/2019-01-13-giving-programmers-what-they-want.html
