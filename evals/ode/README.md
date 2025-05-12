# An ODE Solution

This benchmark is adapted from the `an_ode` benchmark from [cmpad][], for which
the [original documentation][] is also available.

The problem is to approximately solve an ODE of form

```math
\begin{array}{llll}
y_i '(t) & = & x_0                          & \mbox{for} \; i = 0 \\
y_i '(t) & = & \sum_{j=1}^i x_j y_{i-1} (t) & \mbox{for} \; i > 0  \\
\end{array}
```

with initial value

```math
y_i (0) = 0  \; \mbox{for all} \; i
```

using Runge-Kutta with $s$ steps.

The differentiated problem is to compute the gradient of the last element of the
primal output, relative to all inputs. That is, both the primal function and the
differentiated function accept a vector of $n$ elements and produce a vector of
$n$ elements (in addition to a parameter $s$, which is not differentiable).

## Protocol

The evaluation protocol is specified in terms of [TypeScript][] types and
references [types defined in the GradBench protocol description][protocol].

### Inputs

The eval sends a leading `DefineMessage` followed by `EvaluateMessages`. The
`input` field of any `EvaluateMessage` will be an instance of the `ODEInput`
type defined below. The `function` field will one of the strings `"primal"` or
`"gradient"`.

```typescript
interface ODEInput extends Runs {
  x: double[];
  s: int;
}
```

Because the input extends `Runs`, the tool is expected to run the function some
number of times. It should include one timing entry with the name `"evaluate"`
for each time it ran the function.

### Outputs

A tool must respond to an `EvaluateMessage` with an `EvaluateResponse`. The type
of the `output` field in the `EvaluateResponse` is in both cases a vector of the
same size as `x`:

```typescript
type PrimalOutput = double[];
type GradientOutput = double[];
```

## Commentary

This eval is best implemented with reverse mode, and the main challenge for this
eval is the sequential loop with $s$ steps, which requires the AD tool to
maintain a tape. The [lstm][] eval is a more complicated problem that exercises
roughly the same parts of AD.

### Parallel execution

The `primal` function is easily parallelised, and this has been done with OpenMP
in [ode.hpp][]. This does not show any speedup on our workloads, however.

[cmpad]: https://github.com/bradbell/cmpad
[original documentation]: https://cmpad.readthedocs.io/an_ode.html
[protocol]: /CONTRIBUTING.md#types
[typescript]: https://www.typescriptlang.org/
[lstm]: /evals/lstm
[ode.hpp]: /cpp/gradbench/evals/ode.hpp
