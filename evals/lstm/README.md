# Long Short-Term Memory (LSTM)

This eval is taken from
[ADBench](https://github.com/microsoft/ADBench), although it is not
discussed in the ADBench paper. LSTM is a kind of recurrent neural
network architecture popular (at least at one point) in named entity
recognition and part-of-speech tagging. The ADBench variant has a
quirk in that all the weight matrices are diagonal. [ADBench seemed to
have a plan to eventually rename the LSTM
benchmark](https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/ADBench/plot_graphs.py#L89-L92),
but never got around to it. For consistency with ADBench, we have
maintained the name.

## Protocol

The evaluation protocol is specified in terms of [TypeScript][] types
and references [types defined in the GradBench protocol
description](https://github.com/gradbench/gradbench?tab=readme-ov-file#types).

### Inputs

The eval sends a leading `DefineMessage` followed by
`EvaluateMessages`. The `input` field of any `EvaluateMessage` will be
an instance of the `LSTMInput` type defined below. The `function` field
will be either the string `"objective"` or `"jacobian"`.

```typescript
interface LSTMInput {
  main_params: double[][];
  extra_params: double[][];
  state: double[][];
  sequence: double[][];
}
```

### Outputs

A tool must respond to an `EvaluateMessage` with an
`EvaluateResponse`. The type of the `output` field in the
`EvaluateResponse` depends on the `function` field in the
`EvaluateMessage`:

* `"objective"`: `LSTMObjectiveOutput`.
* `"jacobian"`: `LSTMObjectiveOutput`.

```typescript
type LSTMObjectiveOutput = double;
type LSTMJacobianOutput = double[];
```

[typescript]: https://www.typescriptlang.org/