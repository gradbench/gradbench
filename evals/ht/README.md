# Hand Tracking (HT)

This eval is adapted from "Objective HT: Hand Tracking" from section 4.3 of the
[ADBench paper][]. It computes .... It defines a module named `ht`, which
consists of two functions `objective` and `jacobian`, both of which take the
same input:

```typescript
import type { Float, Int, Runs } from "gradbench";

interface Model {
  bone_count: Int;
  bone_names: string[];
  parents: Int[];
  base_relatives: Float[][][];
  inverse_base_absolutes: Float[][][];
  base_positions: Float[][];
  weights: Float[][];
  triangles: Int[][];
  is_mirrored: boolean;
}

interface Data {
  model: Model;
  correspondences: Int[];
  points: Float[][];
}

interface Input extends Runs {
  theta: Float[];
  data: Data;
  us: Float[];
}

export namespace ht {
  function objective(input: Input): Float[];
  function jacobian(input: Input): Float[][];
}
```

## Definition

We have $`N \in \mathbb{N}`$ measured points. Let's call the number of triangles
$`T \in \mathbb{N}`$, and the number of bones $`B = 22`$.

- `theta` is $`\boldsymbol{p} \in \mathbb{R}^{26}`$.
  - $`3`$ parameters for global translation.
  - $`3`$ parameters for global rotation parametrized using angle-axis
    representation.
  - $`4`$ angles for every finger.
- `points` is a column-major encoding of the matrix
  $`\mathbf{Y} \in \mathbb{R}^{3 \times N}`$.
- `correspondences` is $`\{1, \dots, T\}^N`$
- `us` is a column-major encoding of the matrix
  $`\mathbf{U} \in \mathbb{R}^{2 \times N}`$.

Then, for the model:

- `bone_count` is $`B`$.
- `bone_names` isn't used in the computation; it's just an array of length
  $`B`$.
- `parents` is in $`(\{-1\} \cup \{1, \dots, B\})^B`$.
- `base_relatives` is, for each bone $`i \in \{1, \dots, B\}`$, a matrix in
  $`\mathbb{R}^{4 \times 4}`$.
- `inverse_base_absolutes` is similarly, for each bone
  $`i \in \{1, \dots, B\}`$, a matrix in $`\mathbb{R}^{4 \times 4}`$.
- `base_positions` is a column-major encoding of a matrix in
  $`\mathbb{R}^{4 \times M}`$.
- `weights` is a column-major encoding of the matrix
  $`\mathbf{W} \in \mathbb{R}^{B \times M}`$.
- `triangles` is in $`\{1, \dots, M\}^T`$.
- `is_mirrored` seems to always be false.

For each bone $`i \in \{1, \dots, B\}`$, we make a transformation matrix
$`\mathbf{T}_i \in \mathbb{R}^{4 \times 4}`$

## Commentary

### Parallel execution

The reference implementation of `objective` in the [C++ implementation][] has
been partially parallelised using OpenMP, although the impact is minor on most
workloads. More work could possibly make it perform better.

Parallelisation of `jacobian` is trivial, as the multiple passes can be executed
independently.

[adbench paper]: https://arxiv.org/abs/1807.10129
[c++ implementation]: /cpp/gradbench/evals/ht.hpp
