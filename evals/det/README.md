# Determinant Using Expansion by Minors

This benchmark is adapted from the `det_by_minor` benchmark from
[cmpad][], for which the [original documentation][] is also available.

The problem is to compute the determinant of a square matrix

```math
A : \mathbb{R}^{\ell \times \ell}.
```

This is done through [expansion by minors][], which is not an
appropriate algorithm for any but the smallest matrices, but does
provide a good source of floating point operations.

This eval has two functions:

- `primal` is given $A$ and must compute the determinant of $A$ as above.

- `gradient` is given $A$ and must compute the gradient of the
  determinant of $A$ with respect to $A$.

## Protocol

The evaluation protocol is specified in terms of [TypeScript][] types
and references [types defined in the GradBench protocol
description][protocol].

### Inputs

The eval sends a leading `DefineMessage` followed by
`EvaluateMessages`. The `input` field of any `EvaluateMessage` will be
an instance of the `LLSQInput` type defined below. The `function`
field will one of the strings `"primal"` or `"gradient"`.

```typescript
interface DetByMinorInput extends Runs {
  A: double[];
  ell: int;
}
```

The `A` field contains $A$ in row-major order, and the `ell` field is
the edge size $\ell$. This is strictly speaking redundant, as `ell`
will be the square root of the length of the flat representation.

Because the input extends `Runs`, the tool is expected to run the
function some number of times. It should include one timing entry with
the name `"evaluate"` for each time it ran the function.

### Outputs

The type of the output depends on the function.

```typescript
type PrimalOutput = double;
type GradientOutput = double[];
```

The `GradientOutput` is the gradient in row-major order, its length is
equal to the length of the input field `A`.

## Commentary

This benchmark is best implemented with reverse mode. The main
challenge (for some tools) is that the direct formulation of this
problem is extremely scalar-oriented, and based on recursion. This
makes it difficult or impossible for tools to bundle up the work in
larger chunks, which particularly hinders systems such as PyTorch. One
solution is to refactor the computation to be nonrecursive, and
instead directly compute all of the necessary $O(n!)$ permutations of
multiplications. This is overall more expensive, but is for some tools
the only way to express this problem. See [det.fut][] for an example.

[cmpad]: https://github.com/bradbell/cmpad
[original documentation]: https://cmpad.readthedocs.io/det_by_minor.html
[expansion by minors]: https://mathworld.wolfram.com/DeterminantExpansionbyMinors.html
[protocol]: /CONTRIBUTING.md#types
[typescript]: https://www.typescriptlang.org/
[det.fut]: /tools/futhark/det.fut
