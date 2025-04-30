# Bundle Adjustment (BA)

This eval implements Bundle Adjustment from Microsoft's [ADBench][], based on their Python implementation. Here are links to the [I/O file][io] and [data folder][data] from that original implementation.

## Generation

Files are provided with information to generate the below inputs. 3 values, N, M, and P are used alongside one camera's paremters $c \in \mathbb{R}^{11}$, one point $x \in \mathbb{R}^{3}$, one weight $w \in \mathbb{R}$, and one feature $f \in \mathbb{R}^{2}$. N is the number of cameras, M is the numer of points, and P is the number of observations.

## Description

### Inputs

The data generator returns a dictionary with the following inputs

1. Cams: Set of camera parameters used. The parameters are rotation, camera center, focal length, principal point, and radial distortion. $c$ is duplicated N times to create

   $$cams \in \mathbb{R}^{N \times 11}$$

2. X: Set of projected 3D points used. $x$ is duplicated M times to create

   $$X \in \mathbb{R}^{M \times 3}$$

3. W: Set of weights used to regularize points. $w$ is duplicated P times to create

   $$W \in \mathbb{R}^{P}$$

4. Feats: Set of observed 2D points used to compute the reprojection error of 3D points. $f$ is duplicated P times to create

   $$feats \in \mathbb{R}^{P \times 2}$$

5. Obs: Set of P pairs of indices of camera parameters and points.

   $$obs \in \mathbb{R}^{P \times 2}$$

### Outputs

1. Reproj-Error: Reprojection error of how well the projected 3D structure aligns with the given 2D image.

$$reproj_{error} \in \mathbb{R}^{2P}$$

2. W-Error: Regularization error ensures the weights are confined to a specific range. In this scenario it penalized $w$ if it deviates from 1.

$$w_{error} \in \mathbb{R}^{P}$$

3. Jacobian ($J$): How the reprojection error will change given adjustments to the above inputs.

$$SparseMat \in \mathbb{R}^{(2P + P) \times (11N +3M + P)}$$

> **Example**
>
> If $N = 21$, $M = 11315$, and $P = 36455$, $J$ is a SparseMat with 109365 rows and 70631 columns

## Protocol

The eval sends a leading `DefineMessage` followed by `EvaluateMessages`. The `input` field of any `EvaluateMessage` will be an instance of the `BAInput` interface defined below. The `function` field will be either the string `"objective"` or `"jacobian"`.

### Inputs

```typescript
interface BAInput extends Runs {
  n: int;
  m: int;
  p: int;
  cam: double[];
  x: double[];
  w: number;
  feat: double[];
}
```

The `p` input is a duplication parameter - the tool is expected to
duplicate the input `p` times. The tool must not actually exploit this
duplication to reduce the work.

### Outputs

A tool must respond to an `EvaluateMessage` with an `EvaluateResponse`. The type of the `output` field in the `EvaluateResponse` depends on the `function` field in the `EvaluateMessage`:

- `"objective"`: `BAObjectiveOutput`.
- `"jacobian"`: `BAJacobianOutput`.

```typescript
interface BAObjectiveOutput {
  reproj_err: double[];
  w_err: double[];
}
```

```typescript
interface BAJacobianOutput {
  cols: int[31];
  rows: int[31];
  vals: double[31];
}
```

The `BAJacobianOutput` represents a sparse matrix in [COO][] format,
albeit in the form of three lists instead of a list of triples.
Further, to make the matrix smaller, we undo the factor-`p`
duplication of the input (see above) and store only a subset of the
full lists. Specifically, we transmit only the first 30 elements and
the last element.

Because the input extends `Runs`, the tool is expected to run the function some number of times. It should include one timing entry with the name `"evaluate"` for each time it ran the function.

## Commentary

The actual objective function in this benchmark is not particularly
challenging, and is mostly a bunch of scalar control flow with a
bounded control flow graph. The annoying part is that the result must
be reported as a particular encoding of a sparse matrix. This is not
so difficult if you are implementing this eval in C++ - use the
provided data structure in [ba.hpp][] and look at how
[tools/manual/run_ba.cpp] does it. It is a lot more annoying if you
are using another language, and _particularly_ if you want to compute
the sparse Jacobian in parallel. See [tools/futhark/ba.fut][] for how
to do this.

### Parallel execution

The `objective` function contains parallelisable loops of size `p`.
These are parallelised with OpenMP in [ba.hpp][].

For `jacobian`, it is straightforward to parallelise the computation
of the nonzero blocks, but it is tricky (but doable) to assemble the
sparse representation of the Jacobian in parallel.

[adbench]: https://github.com/microsoft/ADBench/tree/38cb7931303a830c3700ca36ba9520868327ac87
[data]: https://github.com/microsoft/ADBench/tree/38cb7931303a830c3700ca36ba9520868327ac87/data/ba
[io]: https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/python/shared/BAData.py
[typescript]: https://www.typescriptlang.org/
[COO]: https://en.wikipedia.org/wiki/Sparse_matrix#Coordinate_list_(COO)
[ba.hpp]: /cpp/gradbench/evals/ba.hpp
[tools/manual/run_ba.cpp]: /tools/manual/run_ba.cpp
[tools/futhark/ba.fut]: /tools/futhark/ba.fut
