# Hand Tracking (HT)

This eval is adapted from "Objective HT: Hand Tracking" from section 4.3 of the
[ADBench paper][]. It computes the residuals and Jacobian for a hand tracking
objective based on [linear blend skinning][] and [barycentric coordinates][].
It defines a module named `ht`, which consists of two functions `objective` and
`jacobian`, both of which take the same input:

```typescript
import type { Float, Int, Runs } from "gradbench";

interface Model {
  /** Number of bones. */
  bone_count: Int;

  /** Bone names, length = bone_count. */
  bone_names: string[];

  /** Parent indices for each bone, length = bone_count. */
  parents: Int[];

  /** Base relative transforms, one 4x4 matrix per bone. */
  base_relatives: Float[][][];

  /** Inverse base absolute transforms, one 4x4 matrix per bone. */
  inverse_base_absolutes: Float[][][];

  /** Homogeneous base positions, one 4-vector per vertex. */
  base_positions: Float[][];

  /** Skinning weights, one weight per vertex/bone pair. */
  weights: Float[][];

  /** Triangle indices, one 3-vector per triangle. */
  triangles: Int[][];

  /** Whether to mirror along the X axis. */
  is_mirrored: boolean;
}

interface Data {
  /** Hand model. */
  model: Model;

  /** Correspondence indices. */
  correspondences: Int[];

  /** Measured points. */
  points: Float[][];
}

interface Input extends Runs {
  /** Pose parameters. */
  theta: Float[];

  /** Hand model and data. */
  data: Data;

  /** Barycentric coordinates for correspondences (empty for simple case). */
  us: Float[][];
}

export namespace ht {
  /** Compute residuals. */
  function objective(input: Input): Float[];

  /** Compute Jacobian of residuals. */
  function jacobian(input: Input): Float[][];
}
```

## Definition

The hand model contains $`M`$ vertices and $`T`$ triangles, with
`bone_count = B` (in the provided datasets, $`B = 22`$). The input arrays are
[row-major][] encodings, and indices are 0-based. We write
$`\mathbf{X} \in \mathbb{R}^{M \times 4}`$ for the homogeneous base positions,
$`\mathbf{W} \in \mathbb{R}^{M \times B}`$ for the skinning weights,
$`\mathbf{Y} \in \mathbb{R}^{N \times 3}`$ for the measured points, and
$`\mathbf{U} \in \mathbb{R}^{N \times 2}`$ for the barycentric coordinates. The
correspondence list has length $`N`$.

The pose vector is $`\boldsymbol{p} = \texttt{theta} \in \mathbb{R}^{26}`$:

1. $`p_{0..2}`$ are a global [angle-axis][] rotation vector $`\boldsymbol{\omega}`$.
2. $`p_{3..5}`$ are global translation $`\boldsymbol{t} \in \mathbb{R}^3`$.
3. The remaining 20 entries provide per-finger joint angles, four per finger.
   The first joint of each finger uses two angles and the remaining joints use
   one angle each, all interpreted as [Euler angles][] in $`xzy`$ order.

For each bone $`i \in \{0, \dots, B-1\}`$, we construct a local rotation matrix
$`\mathbf{R}_i \in \mathbb{R}^{3 \times 3}`$ from its Euler angles. We embed
this in a homogeneous transform and combine it with the model's base transform
to get the relative transform

```math
\mathbf{A}_i = \mathbf{R}^{\text{base}}_i
\begin{bmatrix}
  \mathbf{R}_i & \mathbf{0} \\
  \mathbf{0}^\top & 1
\end{bmatrix}
\quad\text{where}\quad
\mathbf{R}^{\text{base}}_i = \texttt{base\_relatives}[i].
```

The absolute transforms are computed via the parent chain

```math
\mathbf{T}_i =
\begin{cases}
  \mathbf{A}_i & \text{if } \texttt{parents}[i] = -1 \\
  \mathbf{T}_{\texttt{parents}[i]} \mathbf{A}_i & \text{otherwise}
\end{cases}
```

and then adjusted by the inverse base absolutes

```math
\mathbf{S}_i = \mathbf{T}_i \, \mathbf{C}_i
\quad\text{where}\quad
\mathbf{C}_i = \texttt{inverse\_base\_absolutes}[i].
```

Let $`\bar{\boldsymbol{x}}_j \in \mathbb{R}^4`$ denote the $`j`$-th row of
$`\mathbf{X}`$. The skinned vertex positions are

```math
\boldsymbol{v}_j = \sum_{i=1}^B w_{j,i} \, (\mathbf{S}_i \bar{\boldsymbol{x}}_j)_{1:3}
```

where $`w_{j,i}`$ is the weight from $`\mathbf{W}`$. If `is_mirrored` is true,
the $`x`$ coordinate of every $`\boldsymbol{v}_j`$ is negated. Then the global
transform is applied using the [angle-axis][] rotation matrix
$`\mathbf{R}(\boldsymbol{\omega})`$:

```math
\boldsymbol{v}'_j = \mathbf{R}(\boldsymbol{\omega}) \boldsymbol{v}_j + \boldsymbol{t}.
```

Finally, the predicted correspondence point depends on `us`:

1. **Simple case (`us` empty):** `correspondences` contains vertex indices
   $`c_q`$, and $`\boldsymbol{y}'_q = \boldsymbol{v}'_{c_q}`$.
2. **Complicated case (`us` provided):** `correspondences` contains triangle
   indices, so for triangle $`(i,j,k)`$ and barycentric coordinates
   $`\boldsymbol{u}_q = (u_{q,1}, u_{q,2})`$,

```math
\boldsymbol{y}'_q =
u_{q,1} \boldsymbol{v}'_i +
u_{q,2} \boldsymbol{v}'_j +
(1 - u_{q,1} - u_{q,2}) \boldsymbol{v}'_k.
```

The residuals are

```math
\boldsymbol{e}_q = \boldsymbol{y}_q - \boldsymbol{y}'_q
```

and `objective` returns the row-major encoding of the matrix
$`\mathbf{E} \in \mathbb{R}^{N \times 3}`$ whose rows are $`\boldsymbol{e}_q`$.

`jacobian` returns a matrix with $`3N`$ rows. In the simple case, it has 26
columns (ordered as $`p_0, \dots, p_{25}`$) and contains
$`\partial \mathbf{E} / \partial \boldsymbol{p}`$. In the
complicated case, it has 28 columns: the first two columns correspond to
$`\partial \boldsymbol{e}_q / \partial u_{q,1}`$ and
$`\partial \boldsymbol{e}_q / \partial u_{q,2}`$ for each correspondence,
followed by the 26 columns for $`\boldsymbol{p}`$. This uses the fact that each
residual depends only on its own barycentric coordinates.

## Commentary

### Parallel execution

The reference implementation of `objective` in the [C++ implementation][] has
been partially parallelised using OpenMP, although the impact is minor on most
workloads. More work could possibly make it perform better.

Parallelisation of `jacobian` is trivial, as the multiple passes can be executed
independently.

[adbench paper]: https://arxiv.org/abs/1807.10129
[angle-axis]: https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation
[barycentric coordinates]: https://en.wikipedia.org/wiki/Barycentric_coordinate_system
[euler angles]: https://en.wikipedia.org/wiki/Euler_angles
[linear blend skinning]: https://en.wikipedia.org/wiki/Skeletal_animation#Skinning
[row-major]: https://en.wikipedia.org/wiki/Row-_and_column-major_order
[c++ implementation]: /cpp/gradbench/evals/ht.hpp
