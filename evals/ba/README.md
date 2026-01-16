# Bundle Adjustment (BA)

This eval is adapted from "Objective BA: Bundle Adjustment" from section 4.2 of
the [ADBench paper][], with a more complete, self-contained specification. It
computes the residuals and Jacobian for a sparse bundle adjustment problem with
radial distortion and a per-observation weight regularizer. It defines a module
named `ba`, which consists of two functions `objective` and `jacobian`, both of
which take the same input:

```typescript
import type { Float, Int, Runs } from "gradbench";

/** The full input. */
interface Input extends Runs {
  /** Number of cameras. */
  n: Int;

  /** Number of points. */
  m: Int;

  /** Number of observations. */
  p: Int;

  /** Camera parameters. */
  cam: Float[];

  /** 3D point. */
  x: Float[];

  /** Weight. */
  w: Float;

  /** Observed image coordinates. */
  feat: Float[];
}

interface ObjectiveOutput {
  reproj_error: { elements: Float[]; repeated: Int };
  w_err: { element: Float; repeated: Int };
}

interface JacobianOutput {
  rows: Int[];
  cols: Int[];
  vals: Float[];
}

export namespace ba {
  /** Compute the residuals. */
  function objective(input: Input): ObjectiveOutput;

  /** Compute the Jacobian of the residuals. */
  function jacobian(input: Input): JacobianOutput;
}
```

## Definition

We define a single camera using the parameter vector
$`\boldsymbol{p} \in \mathbb{R}^{11}`$ with layout

```math
\boldsymbol{p} = [r_1, r_2, r_3, C_1, C_2, C_3, f, u_0, v_0, k_1, k_2],
```

where $`\boldsymbol{r} = (r_1, r_2, r_3)`$ is the [axis-angle][] rotation,
$`\boldsymbol{C} = (C_1, C_2, C_3)`$ is the camera center, $`f`$ is the focal
length, $`\boldsymbol{x}_0 = (u_0, v_0)`$ is the principal point, and
$`\boldsymbol{\kappa} = (k_1, k_2)`$ are the radial distortion parameters. The
inputs `cam`, `x`, `w`, and `feat` are a single camera, a single 3D point, a
single weight, and a single 2D feature. The eval expands them into full arrays
as follows:

- `cams` is $`\mathbb{R}^{N \times 11}`$, formed by duplicating `cam` `n` times.
- `X` is $`\mathbb{R}^{M \times 3}`$, formed by duplicating `x` `m` times.
- `w` is $`\mathbb{R}^{P}`$, formed by duplicating the scalar `w` `p` times.
- `feats` is $`\mathbb{R}^{P \times 2}`$, formed by duplicating `feat` `p`
  times.
- `obs` is $`\mathbb{N}^{P \times 2}`$ where the $`i`$-th entry is
  $`(i \bmod n, i \bmod m)`$.

Each observation $`i`$ uses camera $`\boldsymbol{p}_{c(i)}`$, point
$`\boldsymbol{X}_{x(i)}`$, weight $`w_i`$, and feature $`\boldsymbol{m}_i`$,
where $`c(i)`$ and $`x(i)`$ are the indices from `obs`. Although the data are
duplicated, tools must still perform the full `p` observations without
exploiting repetition to reduce work.

The projection function is defined by

```math
\operatorname{project}(\boldsymbol{p}, \boldsymbol{X}) =
f \cdot \operatorname{distort}(\boldsymbol{\kappa},
\operatorname{p2e}(\operatorname{rodrigues}(\boldsymbol{r}, \boldsymbol{X} - \boldsymbol{C}))) + \boldsymbol{x}_0,
```

with helper functions

```math
\operatorname{p2e}(\boldsymbol{u}) = \frac{u_{1:2}}{u_3}
```

and

```math
\operatorname{distort}(\boldsymbol{\kappa}, \boldsymbol{u}) =
\boldsymbol{u} (1 + k_1 \|\boldsymbol{u}\|^2 + k_2 \|\boldsymbol{u}\|^4).
```

The Rodrigues rotation used above is

```math
\operatorname{rodrigues}(\boldsymbol{r}, \boldsymbol{x}) =
\boldsymbol{x} \cos \theta + (\boldsymbol{v} \times \boldsymbol{x}) \sin \theta + \boldsymbol{v} (\boldsymbol{v}^\top \boldsymbol{x}) (1 - \cos \theta),
```

where $`\theta = \|\boldsymbol{r}\|`$ and
$`\boldsymbol{v} = \boldsymbol{r} / \theta`$ for $`\theta \neq 0`$. For
$`\theta = 0`$, the eval uses the first-order approximation
$`\operatorname{rodrigues}(\boldsymbol{r}, \boldsymbol{x}) = \boldsymbol{x} + \boldsymbol{r} \times \boldsymbol{x}`$.

The residuals for observation $`i`$ are

```math
\boldsymbol{e}_i =
\begin{bmatrix}
w_i (\operatorname{project}(\boldsymbol{p}_{c(i)}, \boldsymbol{X}_{x(i)}) - \boldsymbol{m}_i) \\
1 - w_i^2
\end{bmatrix},
```

and the `objective` function returns all reprojection residuals
$`\text{reproj\_error} \in \mathbb{R}^{2P}`$ and weight residuals
$`\text{w\_err} \in \mathbb{R}^P`$ without summing or squaring them.

## Jacobian structure

Let $`J`$ be the Jacobian of the full residual vector
$`[\boldsymbol{e}_1^\top, \dots, \boldsymbol{e}_P^\top]^\top`$ with respect to
all independent variables $`\{\boldsymbol{p}_i\}_{i=1}^N`$,
$`\{\boldsymbol{X}_j\}_{j=1}^M`$, and $`\{w_i\}_{i=1}^P`$. Its shape is

```math
J \in \mathbb{R}^{(2P + P) \times (11N + 3M + P)}.
```

Column ordering is: all camera parameters (11 per camera), then all 3D points (3
per point), then all weights (1 per observation). Row ordering is: all
reprojection rows (2 per observation), then all weight rows (1 per observation).
Each reprojection row depends only on one camera, one point, and one weight, so
each of those rows has exactly $`11 + 3 + 1 = 15`$ non-zero entries. The weight
rows have a single non-zero entry:

```math
\frac{\partial}{\partial w_i}(1 - w_i^2) = -2 w_i.
```

The full sparse Jacobian is represented in a compressed-row format where `rows`
has length $`(2P + P + 1)`$ and `cols`/`vals` have length
$`(11 + 3 + 1) \cdot 2P + P`$. For each row $`r`$, the non-zero columns are
stored in `cols[rows[r] .. rows[r + 1] - 1]` with corresponding values from
`vals`.

## Output encoding

Because the input data are formed by duplication, the output is transmitted in a
compressed form:

- `objective` returns the first two reprojection residuals in
  `reproj_error.elements`, along with `reproj_error.repeated = p`; the full
  `reproj_error` vector is obtained by repeating those two elements `p` times in
  order. Similarly, `w_err.element` is the first weight residual with
  `w_err.repeated = p`.
- `jacobian` returns `rows`, `cols`, and `vals` each with length 31, containing
  the first 30 elements and the last element of the full arrays. Equivalently,
  if `A` is one of the full arrays, the output is
  `A[0..29] ++ [A[A.length - 1]]`.

## Commentary

The objective function is straightforward, but the result must be reported as a
particular compressed sparse encoding. If you are implementing this eval in C++,
use the data structure in [ba.hpp][]. For other languages, it can be handy to
compare against the manual implementation in [tools/manual/ba.cpp][]. See
[tools/futhark/ba.fut][] for a parallel-friendly approach.

### Parallel execution

The `objective` function contains parallelisable loops of size `p`. These are
parallelised with OpenMP in [ba.hpp][].

For `jacobian`, it is straightforward to parallelise the computation of the
nonzero blocks, but it is tricky (but doable) to assemble the sparse
representation of the Jacobian in parallel.

[adbench paper]: https://arxiv.org/abs/1807.10129
[axis-angle]: https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation
[ba.hpp]: /cpp/gradbench/evals/ba.hpp
[tools/futhark/ba.fut]: /tools/futhark/ba.fut
[tools/manual/ba.cpp]: /tools/manual/ba.cpp
