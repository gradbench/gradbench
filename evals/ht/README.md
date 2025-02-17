# Hand Tracking (HT)

This eval is taken from
[ADBench](https://github.com/microsoft/ADBench). See [the ADBench
paper](https://arxiv.org/abs/1807.10129) for details on the underlying
problem. The data files are also taken directly from ADBench. ADBench
also sometimes refers to the HT problem as "HAND", but GradBench seeks
to use the name "HT" consistently.

ADBench distinguishes explicitly between "complicated" and "simple"
inputs, which differ in whether the `us` vector (see below) is empty.
In GradBench there is no such distinction, and the "simple" case is
simply identified by a zero-length `us`.

## Protocol

The protocol is specified in terms of [TypeScript][] types and
references [types defined in the GradBench protocol
description](https://github.com/gradbench/gradbench?tab=readme-ov-file#types).

### Inputs

The eval sends a leading `DefineMessage` followed by
`EvaluateMessages`. The `input` field of any `EvaluateMessage` will be
an instance of the `HTInput` type defined below. The `function` field
will be either the string `"objective"` or `"jacobian"`.


```typescript
interface HTModel {
  bone_count: int;
  bone_names: string[];
  parents: int[];
  base_relatives: double[][][];
  inverse_base_absolutes: double[][][];
  base_positions: double[][];
  weights: double[][];
  triangles: int[][];
  is_mirrored: bool;
}

interface HTData {
  model: HTModel;
  correspondences: int[];
  points: double[][];
}

interface HTInput {
  theta: double[];
  data: HTData;
  us: double[];
}
```

### Outputs

A tool must respond to an `EvaluateMessage` with an
`EvaluateResponse`. The type of the `output` field in the
`EvaluateResponse` depends on the `function` field in the
`EvaluateMessage`:

* `"objective"`: `HTObjectiveOutput`.
* `"jacobian"`: `HTJacobianOutput`.

```typescript
type HTObjectiveOutput = double[];
type HTJacobianOutput = double[][];
```

[typescript]: https://www.typescriptlang.org/
