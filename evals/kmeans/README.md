# _k_-means clustering

_k_-means clustering is an algorithm for partitioning $n$ $d$-dimensional
observations into $k$ clusters, such that we minimise the sum of distances from
each point to the centroid of its cluster. This can be done using Newton's
Method, which requires computing the Hessian of an appropriate cost function.
The Hessian also happens to be sparse - it only has nonzero elements along the
diagonal.

A discussion of this approach can be found in [Convergence Properties of the
K-Means Algorithms][paper] by Léon Bottou and Yoshua Bengio (NIPS 1994).

## Idea

Given $n$ points $P$ and $k$ centroids $C$, the objective function is

```math
f(P, C) = \sum_i \text{min}_j(|C_j-P_i|^2)
```

and we must find the $C$ that minimises it.

First we find the derivative with respect to $C$, which produces a $k$-element
gradient (where each element is a $d$-dimensional point), which is our
(single-row) Jacobian $J$.

Then we take the derivative of (the function that computes) $J$, to compute a
Hessian $H$, which has nonzero elements only along the diagonal, meaning $H$ can
be represented as a $k$-element vector of $d$-dimensional points.

We finally compute $J * H^{-1}$, which is the result that must be reported by
the tool for the `"dir"` function. Note that since $H$ is sparse diagonal,
$H^{-1}$ is simply the inverse of each element of $H$.

## Protocol

The evaluation protocol is specified in terms of [TypeScript][] types and
references [types defined in the GradBench protocol description][protocol].

### Inputs

The eval sends a leading `DefineMessage` followed by `EvaluateMessages`. The
`input` field of any `EvaluateMessage` will be an instance of the `KMeansInput`
type defined below. The `function` field will one of the strings `"cost"` or
`"dir"`.

```typescript
interface KMeansInput extends Runs {
  points: double[][];
  centroids: double[][];
}
```

Here `points` is $P$ and `centroids` is $C$. Each element `points[i]` or
`centroids[i]` denotes a $d$-dimensional point (i.e., "row major order"). To
ensure an invertible Hessian, it is guaranteed that each centroid has at least
one point for which it is the closest centroid.

### Outputs

A tool must respond to an `EvaluateMessage` with an `EvaluateResponse`. The type
of the `output` field in the `EvaluateResponse` depends on the `function`:

```typescript
type CostOutput = number;
type DirOutput = double[][];
```

Because the input extends `Runs`, the tool is expected to run the function some
number of times. It should include one timing entry with the name `"evaluate"`
for each time it ran the function.

## Commentary

Despite the quite simple objective function, this benchmark exercises a number
of interesting things:

1. Efficiently computing a _sparse_ Hessian (just the diagonal). This means that
   a tool that can compute the Hessian, but only the entire thing, would be at a
   disadvantage.

2. Computing the Jacobian along with the Hessian. It is most efficient to
   compute both at the same time (reusing work), but a tool might not allow
   this.

3. Computing the adjoint of the minimum operation in the primal function may not
   be entirely straightforward, as it is "sparse", but it is important for
   performance.

### Parallel execution

The `objective` function is essentially a parallel reduction over the _n_ input
points, which can be parallelised very efficiently. The reference implementation
in [kmeans.hpp][] has been parallelised with OpenMP. Similarly, `jacobian` is a
kind of parallel histogram, which also admits efficient parallel execution.

[paper]:
  https://proceedings.neurips.cc/paper/1994/hash/a1140a3d0df1c81e24ae954d935e8926-Abstract.html
[protocol]: /CONTRIBUTING.md#types
[typescript]: https://www.typescriptlang.org/
[kmeans.hpp]: /cpp/gradbench/evals/kmeans.hpp
