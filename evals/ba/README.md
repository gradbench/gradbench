# Bundle Adjustment (BA)

Information on the BA equation from Microsoft's [ADBench](https://github.com/microsoft/ADBench/tree/38cb7931303a830c3700ca36ba9520868327ac87) based on their python implementation.

Link to [I/O file](https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/python/shared/BAData.py) and [data folder](https://github.com/microsoft/ADBench/tree/38cb7931303a830c3700ca36ba9520868327ac87/data/ba).

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

The eval sends a leading `DefineMessage` followed by
`EvaluateMessages`. The `input` field of any `EvaluateMessage` will be
an instance of the `BAInput` interface defined below. The `function` field
will be either the string `"objective"` or `"jacobian"`.

[typescript]: https://www.typescriptlang.org/

### Inputs

```typescript
interface BAInput {
  n: int;
  m: int;
  p: int;
  cam: double[];
  x: double[];
  w: number;
  feat: double[];
}
```

### Outputs

A tool must respond to an `EvaluateMessage` with an
`EvaluateResponse`. The type of the `output` field in the
`EvaluateResponse` depends on the `function` field in the
`EvaluateMessage`:

* `"objective"`: `BAObjectiveOutput`.
* `"jacobian"`: `BAObjectiveOutput`.

```typescript
interface BAOutput {
  "cols": int[];
  "rows": int[];
  "vals": double[];
}
```
