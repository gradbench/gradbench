# K-means clustering

K-means clustering is an algorithm for partitioning $n$
$d$-dimensional observations into $k$ clusters, such that we minimise
the sum of distances from each point to the centroid of its cluster.
This can be done using Newton's Method, which requires computing the
Hessian of an appropriate cost function. The Hessian also happens to
be sparse - it only has nonzero elements along the diagonal.

## Idea

Given $n$ points $P$ and $k$ clusters $C$, the objective function is

```math
f(P, C) = \sum_{i<n} \text{min}_{j<k} ||C_j-P_i||
```

and we must find the $C$ that minimises it.

First we find the derivative with respect to $C$, which produces a
$k$-element gradient (where each element is a $d$-dimensional point),
which is our (single-row) Jacobian $J$.

Then we take the derivative of (the function that computes) $J$, to
compute a Hessian $H$, which has nonzero elements only along the
diagonal.

We finally compute $J * H^{-1}$, which is the result that must be
reported by the tool for the `"direction"` function. Note that since
$H$ is sparse, $H^{-1}$ is simply the inverse of each element of $H$.

## Protocol

The evaluation protocol is specified in terms of [TypeScript][] types
and references [types defined in the GradBench protocol
description](https://github.com/gradbench/gradbench?tab=readme-ov-file#types).

### Inputs

The eval sends a leading `DefineMessage` followed by
`EvaluateMessages`. The `input` field of any `EvaluateMessage` will be
an instance of the `KMeansInput` type defined below. The `function`
field will one of the strings `"cost"` or `"dir"`.

```typescript
type KMeansInput (number, double[][]);
```

The number is $k$, and the array is $P$. The clusters $C$ are simply
the last $k$ elements of $P$.

### Outputs

A tool must respond to an `EvaluateMessage` with an
`EvaluateResponse`. The type of the `output` field in the
`EvaluateResponse` depends on the `function`:

```typescript
type CostOutput = number;
type DirOutput = double[][];
```
