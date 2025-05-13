# Gaussian Mixture Model Fitting (GMM)

This eval is adapted from "Objective GMM: Gaussian Mixture Model Fitting" from
section 4.1 of the [ADBench paper][]; it computes the [gradient][] of the
[logarithm][] of the [posterior probability][] for a [multivariate Gaussian][]
[mixture model][] (GMM) with a [Wishart][] [prior][] on the [covariance
matrices][]. It defines a module named `gmm`, which consists of two functions
`objective` and `jacobian`, both of which take the same input:

```typescript
import type { Float, Int, Runs } from "gradbench";

/** Independent variables for which the gradient must be computed. */
interface Independent {
  /** Means. */
  mu: Float[][];

  /** Logarithms of diagonal part for making inverse covariance matrices. */
  q: Float[][];

  /** Jagged lower-triangular part for making inverse covariance matrices. */
  l: Float[][][];

  /** Parametrization for weights. */
  alpha: Float[];
}

/** The full input. */
interface Input extends Runs, Independent {
  /** Dimension of the space. */
  d: Int;

  /** Number of means. */
  k: Int;

  /** Number of points. */
  n: Int;

  /** Data points. */
  x: Float[][];

  /** Additional degrees of freedom in Wishart prior. */
  m: Int;

  /** Inverse standard deviation in Wishart prior. */
  gamma: Float;
}

export namespace gmm {
  /** Compute the log posterior probability. */
  function objective(input: Input): Float;

  /** Compute the gradient of the log posterior probability. */
  function jacobian(input: Input): Independent;
}
```

## Definition

The `Input` field `k` is the number of mixture components $`K \in \mathbb{N}`$,
and `n` is the number of observations $`N \in \mathbb{N}`$. The mixture weights
$`\boldsymbol{\phi} \in [0, 1]^K`$ are computed from `alpha` representing
$`\boldsymbol{\alpha} \in \mathbb{R}^K`$, via the formula

```math
\phi_k = \frac{\exp(\alpha_k)}{\sum_{k'=1}^K \exp(\alpha_{k'})}
```

which ensures that $`\sum_{k=1}^K \phi_k = 1`$. These represent the probability
for each mixture component, so conceptually there is a vector
$`\boldsymbol{z} \in \{1, \dots, K\}^N`$ where $`z_i`$ is the component of
observation $`i`$; but $`\boldsymbol{z}`$ are [latent variables][] and do not
appear in the computation.

Because the model is multivariate, the observations `x` are a [row-major][]
encoding of the matrix $`\mathbf{X} \in \mathbb{R}^{N \times D}`$ where `d` is
the dimension $`D \in \mathbb{N}`$ of the space. Because the distribution is
Gaussian, `mu` is a row-major encoding of the matrix
$`\mathbf{M} \in \mathbb{R}^{K \times D}`$ giving the mean
$`\boldsymbol{\mu}_k \in \mathbb{R}^D`$ for the component with index
$`k \in \{1, \dots, K\}`$.

Conceptually, that component also has a [positive-definite][] covariance matrix
$`\mathbf{\Sigma}_k \in \mathbb{R}^{D \times D}`$. However, the covariance
matrix is not directly used in the computation; only its inverse is used. The
`q` and `l` fields parametrize these inverses of the covariance matrices. If we
represent the zero-indexed value $`k - 1`$ in code as `k: Int`, then the
elements `q[k]` and `l[k]` represent a vector
$`\boldsymbol{q}_k \in \mathbb{R}^D`$ and a [strictly lower triangular][] matrix
$`\mathbf{L}_k \in \mathbb{R}^{D \times D}`$, respectively. Note that `l[k]`
does not include the elements of $`\mathbf{L}_k`$ that are guaranteed to be zero
due to it being strictly lower triangular. That is, `l[k][0]` is empty;
`l[k][1]` has one element $`l_{k,2,1}`$; `l[k][2]` has two elements
$`l_{k,3,1}`$ and $`l_{k,3,2}`$; and so on. From these, we construct the lower
triangular matrix

```math
Q(\boldsymbol{q}_k, \mathbf{L}_k) = \begin{bmatrix}
  \exp(q_{k,1}) & 0 & \cdots & 0 \\
  l_{k,2,1} & \exp(q_{k,2}) & \cdots & 0 \\
  \vdots & \vdots & \ddots & \vdots \\
  l_{k,D,1} & l_{k,D,2} & \cdots & \exp(q_{k,D})
\end{bmatrix} \in \mathbb{R}^{D \times D}
```

by exponentiating each value of $`\boldsymbol{q}_k`$ to form the diagonal and
then summing with $`\mathbf{L}_k`$. Then we use this to compute the inverse of
the covariance matrix as
$`\mathbf{\Sigma}_k^{-1} = Q(\boldsymbol{q}_k, \mathbf{L}_k)Q(\boldsymbol{q}_k, \mathbf{L}_k)^\top`$.

We will refer to the collection of all the means and covariance matrices
together by the symbol $`\boldsymbol{\theta}`$. Then, since component $`k`$ has
probability $`\phi_k`$ and distribution
$`\mathcal{N}_D(\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)`$, we can write the
likelihood by multiplying across all observations and summing across all mixture
components, to get the overall GMM probability density function

```math
p(\mathbf{X} \mid \boldsymbol{\theta}, \boldsymbol{\phi}) = \prod_{i=1}^N \sum_{k=1}^K \phi_k f_{\mathcal{N}_D(\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}(\boldsymbol{x}_i)
```

using the probability density function for the multivariate normal distribution

```math
f_{\mathcal{N}_D(\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}(\boldsymbol{x}_i) = \frac{\exp(-\frac{1}{2}(\boldsymbol{x}_i - \boldsymbol{\mu}_k)^T \mathbf{\Sigma}_k^{-1} (\boldsymbol{x}_i - \boldsymbol{\mu}_k))}{\sqrt{(2\pi)^D \det(\mathbf{\Sigma}_k)}}.
```

These two formulae are not used directly in the computation, but they will allow
us to define the log posterior function we actually want to compute, which we
will then simplify. Before that, though, we must define our prior on the
covariance matrices. The fields `m` and `gamma` are
$`m \in \mathbb{Z}_{\geq 0}`$ and $`\gamma > 0`$ respectively, which parametrize
the Wishart distribution with probability density function

```math
f_{W_D(\mathbf{V}, n)}(\mathbf{\Sigma}_k) = \frac{\det(\mathbf{\Sigma}_k)^{(n-D-1)/2} \exp(-\frac{1}{2} \text{tr}(\mathbf{V}^{-1} \mathbf{\Sigma}_k))}{2^{(Dn)/2} \det(\mathbf{V})^{n/2} \Gamma_D(\frac{n}{2})}
```

where $`\text{tr}`$ is the [trace][], $`\Gamma_D`$ is the [multivariate gamma
function][], and we choose
$`\mathbf{V} = \frac{1}{\gamma^2} I \in \mathbb{R}^{D \times D}`$ and
$`n = D + m + 1`$. From this, we define our prior to be

```math
p(\boldsymbol{\theta}) = \prod_{k=1}^K f_{W_D(\mathbf{V}, n)}(\mathbf{\Sigma}_k)
```

from which we define the `objective` function to compute the log posterior

```math
F_\mathbf{X}(\boldsymbol{\theta}, \boldsymbol{\phi}) = \log(p(\mathbf{X} \mid \boldsymbol{\theta}, \boldsymbol{\phi}) p(\boldsymbol{\theta}))
```

and the `jacobian` function to compute $`\nabla F_{\mathbf{X}}`$ with respect to
the `Independent` variables that parametrize $`\boldsymbol{\theta}`$ and
$`\boldsymbol{\phi}`$.

## Implementation

To actually compute `objective`, it is typical to first perform some algebraic
simplifications, since the Gaussian and Wishart probability density functions
include [determinants][] and matrix inversions that would be expensive to
compute naively. By observing that

```math
\det(\mathbf{\Sigma}_k) = \frac{1}{\det(Q(\boldsymbol{q}_k, \boldsymbol{L}_k))^2} = \frac{1}{\prod_{j=1}^D \exp(q_{k,j})^2} = \exp\Bigg({-2\sum_{j=1}^D q_{k,j}}\Bigg)
```

we can define

```math
\beta_{i,k} = \alpha_k -\frac{1}{2} \|Q(\boldsymbol{q}_k, \mathbf{L}_k)(\boldsymbol{x}_i - \boldsymbol{\mu}_k)\|^2 + \sum_{j=1}^D q_{k,j}
```

and write

```math
p(\mathbf{X} \mid \boldsymbol{\theta}, \boldsymbol{\phi}) = \prod_{i=1}^N \frac{1}{\sqrt{(2\pi)^D} \sum_{k=1}^K \exp(\alpha_k)} \sum_{k=1}^K \exp(\beta_{i,k}).
```

For the prior, we have

```math
\text{tr}(\mathbf{V}^{-1} \mathbf{\Sigma}_k) = \gamma^2\,\text{tr}(\mathbf{\Sigma}_k) = \frac{\gamma^2}{2} \|Q(\boldsymbol{q}_k, \mathbf{L}_k)\|_F^2
```

and $`\det(\mathbf{V})^{n/2} = \gamma^{-Dn}`$, so

```math
f_{W_D(\mathbf{V}, n)}(\mathbf{\Sigma}_k) = \bigg(\frac{\gamma}{\sqrt{2}}\bigg)^{Dn} \frac{1}{\Gamma_D(\frac{n}{2})} \exp\Bigg(\frac{\gamma^2}{2} \|Q(\boldsymbol{q}_k, \mathbf{L}_k)\|_F^2 - m \sum_{j=1}^D q_{k,j}\Bigg).
```

Then we can define a [helper function][`lse`]
$`\text{logsumexp} : \mathbb{R}^K \to \mathbb{R}`$ which we compute stably by
first computing the maximum of the vector as

```math
\text{logsumexp}(\boldsymbol{v}) = \log \sum_{k=1}^K \exp(v_k) = \max \boldsymbol{v} + \log \sum_{k=1}^K \exp(v_k - \max \boldsymbol{v}),
```

and finally use
$`F_\mathbf{X}(\boldsymbol{\theta}, \boldsymbol{\phi}) = \log p(\mathbf{X} \mid \boldsymbol{\theta}, \boldsymbol{\phi}) + \log p(\boldsymbol{\theta})`$
to push the logarithms down, yielding

```math
\log p(\mathbf{X} \mid \boldsymbol{\theta}, \boldsymbol{\phi}) = -N\bigg(\frac{D \log 2\pi}{2} + \text{logsumexp}(\boldsymbol{\alpha})\bigg) + \sum_{i=1}^N \text{logsumexp}(\boldsymbol{\beta}_i)
```

and

```math
\log p(\boldsymbol{\theta}) = K\bigg(Dn \log \frac{\gamma}{\sqrt{2}} - \log \Gamma_D\Big(\frac{n}{2}\Big)\bigg) + \frac{\gamma^2}{2} \sum_{k=1}^K \|Q(\boldsymbol{q}_k, \mathbf{L}_k)\|_F^2 - m \sum_{k=1}^K \sum_{j=1}^D q_{k,j}.
```

## Commentary

This eval is straightforward to implement from an AD perspective, as it computes
a dense gradient of a scalar-valued function. The objective function can be
easily expressed in a scalar way (see the [C++ implementation][cpp] or the
[Futhark implementation][futhark]), or through linear algebra operations (see
the [PyTorch implementation][pytorch]). An even simpler eval with the same
properties is [`llsq`][], which you might consider implementing first. After
implementing `gmm`, implementing [`lstm`][] or [`ode`][] should not be so
difficult.

### Parallel execution

This eval is straightforward to parallelise. The [C++ implementation][cpp] has
been parallelised with OpenMP.

[adbench paper]: https://arxiv.org/abs/1807.10129
[covariance matrices]: https://en.wikipedia.org/wiki/Covariance_matrix
[cpp]: /cpp/gradbench/evals/gmm.hpp
[determinants]: https://en.wikipedia.org/wiki/Determinant
[futhark]: /tool/futhark/gmm.fut
[gradient]: https://en.wikipedia.org/wiki/Gradient
[latent variables]:
  https://en.wikipedia.org/wiki/Latent_and_observable_variables
[logarithm]: https://en.wikipedia.org/wiki/Logarithm
[mixture model]: https://en.wikipedia.org/wiki/Mixture_model
[multivariate gamma function]:
  https://en.m.wikipedia.org/wiki/Multivariate_gamma_function
[multivariate gaussian]:
  https://en.wikipedia.org/wiki/Multivariate_normal_distribution
[positive-definite]: https://en.wikipedia.org/wiki/Definite_matrix
[posterior probability]: https://en.wikipedia.org/wiki/Posterior_probability
[prior]: https://en.wikipedia.org/wiki/Prior_probability
[pytorch]: /python/gradbench/gradbench/tools/pytorch/gmm_objective.py
[row-major]: https://en.wikipedia.org/wiki/Row-_and_column-major_order
[strictly lower triangular]: https://en.wikipedia.org/wiki/Triangular_matrix
[trace]: https://en.m.wikipedia.org/wiki/Trace_(linear_algebra)
[wishart]: https://en.m.wikipedia.org/wiki/Wishart_distribution
[`llsq`]: /evals/llsq
[`lse`]: /evals/lse
[`lstm`]: /evals/lstm
[`ode`]: /evals/ode
