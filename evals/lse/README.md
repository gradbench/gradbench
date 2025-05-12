# LogSumExp

This eval concerns computing the LogSumExp (LSE) function and its gradient. LSE
is a smooth approximation to the function that takes the maximum of a vector of
values $x$. Mathematically, it is defined as

```math
\text{LSE}(x) = \log \sum_i \exp(x_i)
```

However, the exponentials lead to numerical unstability, so in practice we use
the [the LogSumExp trick][] where we first compute

```math
a = \max x
```

and then

```math
\text{LSE}(x) = a + \log \sum_i \exp(x_i-a)
```

This eval has two functions:

- `primal`, where we are given _x_ and must compute _LSE(x)_.

- `gradient`, where we are given _x_ and must compute the gradient of _LSE(x)_.

The LSE function should ideally be implemented in terms of its primitive
elements (computing a maximum, summing a sequence), not by calling a builtin LSE
function with a hand-written derivative.

## Protocol

The evaluation protocol is specified in terms of [TypeScript][] types and
references [types defined in the GradBench protocol description][protocol].

### Inputs

The eval sends a leading `DefineMessage` followed by `EvaluateMessages`. The
`input` field of any `EvaluateMessage` will be an instance of the
`LogSumExpInput` type defined below. The `function` field will one of the
strings `"primal"` or `"gradient"`.

```typescript
interface LogSumExpInput extends Runs {
  x: double[];
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

This is a _very_ simple benchmark, and a good first one to implement.

### Parallel execution

The `primal` function can be implemented as a parallel reduction, and this is
done in [lse.hpp][]. On most workloads, multithreaded execution does not lead to
a significant speedup.

[the LogSumExp trick]: https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
[protocol]: /CONTRIBUTING.md#types
[typescript]: https://www.typescriptlang.org/
[lse.hpp]: /cpp/gradbench/evals/lse.hpp
