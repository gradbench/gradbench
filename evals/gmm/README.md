# Gaussian Mixture Model Fitting (GMM)

This eval implements Gaussian Mixture Model Fitting from Microsoft's [ADBench][], based on their Python implementation. Here are links to [I/O file][io], [data folder][data], and [GMM Data Generator][gen] from that original implementation.

## Generation

To generate files with the below inputs, 3 values are used: D, K, and N. D ranges from $2^1$ to $2^7$ and represents the dimension of the data points and means. K is the number of mixture components (clusters) where K $\in [5,10,25,50,100,200]$. Additionally, GMM can be run with $1000$ or $10000$ data points, where N represents this value. These values/ranges are iterated over to create various datasets.

## Description

### Inputs

The data generator returns a dictionary with the following inputs

1. Alphas ($\alpha$): Mixing components, weights

   $$\alpha \in \mathbb{R}^K$$

2. Means ($M$): Expected centroid points, $\mu_k \in \mathbb{R}^D$

   $$M \in \mathbb{R}^{K \times D}$$

3. Inverse Covariance Factor ($ICF$): Parameteres for the inverse covariance matrix (precision matrix)

   $$ICF \in \mathbb{R}^{K \times (D + \frac{D(D-1)}{2})}$$

4. $X$: Data points being fitted

   $$X \in \mathbb{R}^{N \times D}$$

5. Wishart: Wishart distribution parameters to specify inital beliefs about scale and structure of precision matrcies stored in a tuple

### Outputs

1. Log-Likelihood Value: How well given parameteres fit the given data
2. Gradient ($G$) of Log-Likelihood: How it will change given changes to alphas, means, and ICF $$G \in \mathbb{R}^{K + (K \times D) + (K \times (D + \frac{D(D-1)}{2}))}$$

> **Example**
>
> If $D = 2$ and $K = 5$, $G \in \mathbb{R}^{30}$ meaning the function will return an array of length 30.

## Protocol

The protocol is specified in terms of [TypeScript][] types and references [types defined in the GradBench protocol description][protocol].

### Inputs

The eval sends a leading `DefineMessage` followed by `EvaluateMessages`. The `input` field of any `EvaluateMessage` will be an instance of the `GMMInput` interface defined below. The `function` field will be either the string `"objective"` or `"jacobian"`.

```typescript
interface GMMInput extends Runs {
  d: int;
  k: int;
  n: int;
  alpha: double[];
  means: double[][];
  icf: double[][];
  x: double[][];
  gamma: number;
  m: int;
}
```

The `double[][]` types encode matrices as arrays-of-rows.

### Outputs

A tool must respond to an `EvaluateMessage` with an `EvaluateResponse`. The type of the `output` field in the `EvaluateResponse` depends on the `function` field in the `EvaluateMessage`:

- `"objective"`: `GMMObjectiveOutput`.
- `"jacobian"`: `GMMJacobianOutput`.

```typescript
type GMMObjectiveOutput = double;
type GMMJacobianOutput = double[];
```

The `GMMJacobianOutput` value contains the concatenated gradients for the `alphas`, `means`, and `icf` parameters, in that order.

Because the input extends `Runs`, the tool is expected to run the function some number of times. It should include one timing entry with the name `"evaluate"` for each time it ran the function.

## Commentary

This eval is straightforward to implement from an AD perspective, as
it computes a dense gradient of a scalar-valued function. The
objective function can be easily expressed in a scalar way (see
[gmm.hpp][] or [gmm.fut][]), or through linear algebra operations (see
[gmm_objective.py][]). An even simpler eval with the same properties
is [llsq][], which you might consider implementing first. After
implementing `gmm`, implementing [lstm][] or [ode][] should not be so
difficult.

### Parallel execution

This eval is straightforward to parallelise. The implementation in
[gmm.hpp] has been parallelised with OpenMP.

[adbench]: https://github.com/microsoft/ADBench/tree/38cb7931303a830c3700ca36ba9520868327ac87
[data]: https://github.com/microsoft/ADBench/tree/38cb7931303a830c3700ca36ba9520868327ac87/data/gmm
[gen]: https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/data/gmm/gmm-data-gen.py
[io]: https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/python/shared/GMMData.py
[protocol]: /CONTRIBUTING.md#types
[typescript]: https://www.typescriptlang.org/
[gmm.hpp]: /cpp/gradbench/evals/gmm.hpp
[gmm.fut]: /tool/futhark/gmm.fut
[gmm_objective.py]: /python/gradbench/gradbench/tools/pytorch/gmm_objective.py
[llsq]: /evals/llsq
[lstm]: /evals/lstm
[ode]: /evals/ode
