# Linear Least Squares Objective

This benchmark is adapted from the `llsq_obj` benchmark from [cmpad][], for
which the [original documentation][] is also available.

We are given an objective function defined by

```math
   y(x) = \frac{1}{2} \sum_{i=0}^{n-1} \left(
      s_i - x_0 - x_1 t_i - x_2 t_i^2 - \cdots
   \right)^2
```

where

```math
\begin{array}{ll}
   t_i & = -1 + i * 2 / (n - 1)
   \\
   s_i & = \mathrm{sign} ( t_i )
\end{array}
```

We write $n$ for the size of $t$ and $s$, and $m$ for the size of $x$.

This eval has two functions:

- `primal`, where we are given _x_ and _n_ and must compute _y(x)_.

- `gradient`, where we are given _x_ and _n_ and must compute the gradient of
  _y(x)_.

## Protocol

The evaluation protocol is specified in terms of [TypeScript][] types and
references [types defined in the GradBench protocol description][protocol].

### Inputs

The eval sends a leading `DefineMessage` followed by `EvaluateMessages`. The
`input` field of any `EvaluateMessage` will be an instance of the `LLSQInput`
type defined below. The `function` field will one of the strings `"primal"` or
`"gradient"`.

```typescript
interface LLSQInput extends Runs {
  x: double[];
  n: int;
}
```

Because the input extends `Runs`, the tool is expected to run the function some
number of times. It should include one timing entry with the name `"evaluate"`
for each time it ran the function.

### Outputs

The type of the output depends on the function.

```typescript
type PrimalOutput = double;
type GradientOutput = double[];
```

The size of the `GradientOutput` is equal to the size of the input `x`.

## Commentary

This is a rather simple benchmark, and a good first one to implement. It can
easily be expressed in a vectorised form, which is useful for implementing it in
tools such as [pytorch][].

### Parallel execution

The `primal` function is easily and efficiently parallelised, and this has been
done with OpenMP in [llsq.hpp][]. Speedup is only achieved for the larger
workloads.

[cmpad]: https://github.com/bradbell/cmpad
[original documentation]: https://cmpad.readthedocs.io/llsq_obj.html
[protocol]: /CONTRIBUTING.md#types
[typescript]: https://www.typescriptlang.org/
[pytorch]: /tools/pytorch
[llsq.hpp]: /cpp/gradbench/evals/llsq.hpp
