# K-means clustering

K-means clustering is an algorithm for partitioning $n$
$d$-dimensional observations into $k$ clusters, such that we minimise
the sum of distances from each point to the centroid of its cluster.
This can be done using Newton's Method, which requires computing the
Hessian of an appropriate cost function. The Hessian also happens to
be sparse - it only has nonzero elements along the diagonal.

A discussion of this approach can be found in [Convergence Properties
of the K-Means Algorithms][paper] by Léon Bottou and Yoshua Bengio
(NIPS 1994).

## Idea

Given $n$ points $P$ and $k$ centroids $C$, the objective function is

```math
f(P, C) = \sum_i \text{min}_j(|C_j-P_i|^2)
```

and we must find the $C$ that minimises it.

First we find the derivative with respect to $C$, which produces a
$k$-element gradient (where each element is a $d$-dimensional point),
which is our (single-row) Jacobian $J$.

Then we take the derivative of (the function that computes) $J$, to
compute a Hessian $H$, which has nonzero elements only along the
diagonal, meaning $H$ can be represented as a $k$-element vector of
$d$-dimensional points.

We finally compute $J * H^{-1}$, which is the result that must be
reported by the tool for the `"dir"` function. Note that since
$H$ is sparse, $H^{-1}$ is simply the inverse of each element of $H$.

## Protocol

The evaluation protocol is specified in terms of [TypeScript][] types
and references [types defined in the GradBench protocol
description][protocol].

### Inputs

The eval sends a leading `DefineMessage` followed by
`EvaluateMessages`. The `input` field of any `EvaluateMessage` will be
an instance of the `KMeansInput` type defined below. The `function`
field will one of the strings `"cost"` or `"dir"`.

```typescript
interface KMeansInput extends Runs {
  points: double[][];
  centroids: double[][];
}
```

Here `points` is $P$ and `centroids` is $C$. Each element `points[i]`
or `centroids[i]` denotes a $d$-dimensional point (i.e., "row major
order"). To ensure an invertible Hessian, it is guaranteed that each
centroid has at least one point for which it is the closest centroid.

### Outputs

A tool must respond to an `EvaluateMessage` with an
`EvaluateResponse`. The type of the `output` field in the
`EvaluateResponse` depends on the `function`:

```typescript
type CostOutput = number;
type DirOutput = double[][];
```

[paper]: https://proceedings.neurips.cc/paper/1994/hash/a1140a3d0df1c81e24ae954d935e8926-Abstract.html
[protocol]: https://github.com/gradbench/gradbench?tab=readme-ov-file#types
[typescript]: https://www.typescriptlang.org/
