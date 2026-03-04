# Long Short-Term Memory (LSTM)

This eval is adapted from the LSTM benchmark in [ADBench][]. It defines a module
named `lstm`, which consists of two functions `objective` and `jacobian`, both
of which take the same input:

```typescript
import type { Float, Runs } from "gradbench";

/** The full input. */
interface Input extends Runs {
  /** Layer parameters. */
  main_params: Float[][];

  /** Input/output parameters. */
  extra_params: Float[][];

  /** Initial hidden and cell state for each layer. */
  state: Float[][];

  /** Input sequence. */
  sequence: Float[][];
}

export namespace lstm {
  /** Compute the loss. */
  function objective(input: Input): Float;

  /** Compute the gradient of the loss. */
  function jacobian(input: Input): Float[];
}
```

## Definition

All matrices are encoded as arrays-of-rows, so `Float[][]` should be treated as
row-major. The dimensions are implied by the shapes of the inputs:

- `l` (the layer count) is `main_params.length / 2`.
- `b` (the hidden size) is `main_params[0].length / 4`.
- `c` (the sequence length) is `sequence.length`.

The following shape constraints must hold:

- `main_params` has shape `(2l) x (4b)`.
- `extra_params` has shape `3 x b`.
- `state` has shape `(2l) x b`.
- `sequence` has shape `c x b`.

We define the elementwise sigmoid function as

```math
\sigma(x) = \frac{1}{1 + e^{-x}}.
```

### Parameters

For each layer `i` in `0..l-1`, define weight vectors
`\mathbf{w}_f^{(i)}, \mathbf{w}_i^{(i)}, \mathbf{w}_o^{(i)}, \mathbf{w}_c^{(i)} \in \mathbb{R}^b`
and bias vectors
`\mathbf{b}_f^{(i)}, \mathbf{b}_i^{(i)}, \mathbf{b}_o^{(i)}, \mathbf{b}_c^{(i)} \in \mathbb{R}^b`
by splitting the rows of `main_params`:

- Row `2i` is the concatenation
  `[\mathbf{w}_f^{(i)} \mid \mathbf{w}_i^{(i)} \mid   \mathbf{w}_o^{(i)} \mid \mathbf{w}_c^{(i)}]`.
- Row `2i + 1` is the concatenation
  `[\mathbf{b}_f^{(i)} \mid \mathbf{b}_i^{(i)} \mid   \mathbf{b}_o^{(i)} \mid \mathbf{b}_c^{(i)}]`.

The `extra_params` rows define vectors
`\mathbf{w}_{\text{in}}, \mathbf{w}_{\text{out}}, \mathbf{b}_{\text{out}} \in \mathbb{R}^b`
as

- Row `0` is `\mathbf{w}_{\text{in}}`.
- Row `1` is `\mathbf{w}_{\text{out}}`.
- Row `2` is `\mathbf{b}_{\text{out}}`.

The `state` rows define initial hidden and cell vectors
`\mathbf{h}^{(i)}_0, \mathbf{c}^{(i)}_0 \in \mathbb{R}^b` for each layer:

- Row `2i` is `\mathbf{h}^{(i)}_0`.
- Row `2i + 1` is `\mathbf{c}^{(i)}_0`.

Finally, `sequence[t]` is the input vector `\mathbf{x}_t \in \mathbb{R}^b` at
time step `t`.

### LSTM step

Given a layer input `\mathbf{x} \in \mathbb{R}^b`, hidden state
`\mathbf{h} \in \mathbb{R}^b`, and cell state `\mathbf{c} \in \mathbb{R}^b`, the
update for layer `i` is defined elementwise as

```math
\begin{aligned}
\mathbf{f} &= \sigma(\mathbf{x} \odot \mathbf{w}_f^{(i)} + \mathbf{b}_f^{(i)}) \\
\mathbf{i} &= \sigma(\mathbf{h} \odot \mathbf{w}_i^{(i)} + \mathbf{b}_i^{(i)}) \\
\mathbf{o} &= \sigma(\mathbf{x} \odot \mathbf{w}_o^{(i)} + \mathbf{b}_o^{(i)}) \\
\mathbf{g} &= \tanh(\mathbf{h} \odot \mathbf{w}_c^{(i)} + \mathbf{b}_c^{(i)}) \\
\mathbf{c}' &= \mathbf{c} \odot \mathbf{f} + \mathbf{i} \odot \mathbf{g} \\
\mathbf{h}' &= \mathbf{o} \odot \tanh(\mathbf{c}')
\end{aligned}
```

where `\odot` denotes elementwise multiplication. Note that all weight matrices
are diagonal in this benchmark; no cross-dimension mixing occurs.

### Prediction

Given input `\mathbf{x}_t`, the prediction step uses the extra parameters and
all `l` layers:

```math
\begin{aligned}
\mathbf{z}_0 &= \mathbf{x}_t \odot \mathbf{w}_{\text{in}} \\
(\mathbf{h}^{(0)}_{t+1}, \mathbf{c}^{(0)}_{t+1}) &=
  \text{LSTM}_0(\mathbf{z}_0, \mathbf{h}^{(0)}_t, \mathbf{c}^{(0)}_t) \\
(\mathbf{h}^{(1)}_{t+1}, \mathbf{c}^{(1)}_{t+1}) &=
  \text{LSTM}_1(\mathbf{h}^{(0)}_{t+1}, \mathbf{h}^{(1)}_t, \mathbf{c}^{(1)}_t) \\
\vdots & \\
(\mathbf{h}^{(l-1)}_{t+1}, \mathbf{c}^{(l-1)}_{t+1}) &=
  \text{LSTM}_{l-1}(\mathbf{h}^{(l-2)}_{t+1},
                   \mathbf{h}^{(l-1)}_t, \mathbf{c}^{(l-1)}_t) \\
\mathbf{y}_t &= \mathbf{h}^{(l-1)}_{t+1} \odot \mathbf{w}_{\text{out}} +
  \mathbf{b}_{\text{out}}
\end{aligned}
```

The state is updated at each time step and used for the next prediction.

### Objective

Define a nonstandard log-sum-exp with two extra constant terms:

```math
\operatorname{lse}(\mathbf{y}) =
\log\left(2 + \sum_{i=1}^b e^{y_i}\right).
```

For each `t` in `0..c-2`, compute `\mathbf{y}_t` from `\mathbf{x}_t`, then
normalize as `\mathbf{y}_t - \operatorname{lse}(\mathbf{y}_t)`. Let
`\mathbf{x}_{t+1}` be the next element of the sequence. The loss is

```math
L = -\frac{1}{b(c-1)} \sum_{t=0}^{c-2}
\sum_{i=1}^b x_{t+1,i} \left(y_{t,i} - \operatorname{lse}(\mathbf{y}_t)\right).
```

The `objective` function returns `L`. The `jacobian` function returns the
gradient of `L` with respect to `main_params` and `extra_params`.

## Protocol

The protocol is specified in terms of [TypeScript][] types and references [types
defined in the GradBench protocol description][protocol]. The eval sends a
leading `DefineMessage` followed by `EvaluateMessages`. The `input` field of any
`EvaluateMessage` is an `Input` as defined above. The `function` field is either
`"objective"` or `"jacobian"`.

Because the input extends `Runs`, the tool is expected to run the function some
number of times. It should include one timing entry with the name `"evaluate"`
for each time it ran the function.

### Outputs

A tool must respond to an `EvaluateMessage` with an `EvaluateResponse`. The type
of the `output` field in the `EvaluateResponse` depends on the `function` field:

- `"objective"`: `Float`.
- `"jacobian"`: `Float[]`.

The `jacobian` output is a flat array of length `8lb + 3b`. It is the
concatenation of the row-major gradients of `main_params` followed by
`extra_params`, using the same row order as the inputs.

## Commentary

This eval is similar to [gmm][] in that it asks for the gradient of a
scalar-valued function, and is thus suited for reverse mode. The main additional
challenge is that `lstm` contains a sequential loop over the sequence length
`c`. An even simpler eval with the same properties is [ode][], which you might
consider implementing first.

### Parallel execution

Although the reference implementation in [lstm.hpp][] has been parallelised with
OpenMP, this eval does not benefit much from parallel execution. The reason is
that in contrast to real machine learning models, it is not batched, and so the
amount of independent work is not large. This goes for both `objective` and
`jacobian`.

[adbench]: https://github.com/microsoft/ADBench
[protocol]: /CONTRIBUTING.md#types
[typescript]: https://www.typescriptlang.org/
[gmm]: /evals/gmm
[ode]: /evals/ode
[lstm.hpp]: /cpp/gradbench/evals/lstm.hpp
