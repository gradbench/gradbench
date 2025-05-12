# Gaussian Mixture Model Fitting (GMM)

This eval is "Objective GMM: Gaussian Mixture Model Fitting" from section 4.1 of
the [ADBench paper][]. It defines a module named `gmm`, which consists of two
functions `objective` and `jacobian`, both of which take the same input:

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

  /** Parameterization for weights. */
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

  /** A Wishart prior hyperparameter. */
  m: Int;

  /** A Wishart prior hyperparameter. */
  gamma: Float;
}

export namespace gmm {
  function objective(input: Input): Float;
  function jacobian(input: Input): Independent;
}
```

## Definition

To define these functions, we'll use the same mathematical notation conventions
from section 1 of the ADBench paper:

> $$\text{logsumexp}(\boldsymbol{x} : \mathbb{R}^n) = \log(\text{sum}(\exp(\boldsymbol{x} - \text{max}(\boldsymbol{x})))) + \text{max}(\boldsymbol{x})$$

> In this paper, we use the following notation for variables: scalar $s$ or $S$,
> vector $\boldsymbol{v}$, matrix $\mathbf{M}$, and tensor $\mathsf{T}$. We
> symbolize a concatenation of multiple column vectors
> $\boldsymbol{v}_1, \boldsymbol{v}_2, \dots, \boldsymbol{v}_n$ as a matrix
> $\mathbf{V}$. Similarly, a concatenation of multiple matrices
> $\mathbf{M}_1, \mathbf{M}_2, \dots, \mathbf{M}_m$ as a tensor $\mathsf{M}$.
>
> Special functions are matrix determinant or scalar absolute value $|\cdot|$,
> and Euclidean norm $\|\cdot\|$. Function $\text{logsumexp}$ is always defined
> stably as presented above.

The `Input` field `d` is the dimension $D \in \mathbb{N}$ of the space. The
field `x` is a list of points $\boldsymbol{x}_i \in \mathbb{R}^D$ in that space,
or in other words, a row-major encoding of the matrix
$\mathbf{X} \in \mathbb{R}^{N \times D}$, where the field `n` gives the number
of points $N \in \mathbb{N}$. Similarly, `mu` is a row-major encoding of the
matrix $\mathbf{M} \in \mathbb{R}^{K \times D}$ listing the means
$\boldsymbol{\mu}_k \in \mathbb{R}^D$, where the field `n` gives the number of
means $K \in \mathbb{N}$.

The `q` and `l` fields parameterize the inverses of the covariance matrices.
Note that the encoding here differs slightly from that of the original ADBench
paper, for greater convenience. Let $k \in \{1, \dots, K\}$, so we'll represent
the zero-indexed value $k - 1$ in code as `k: Int`. The elements `q[k]` and
`l[k]` represent the vector $\boldsymbol{q} \in \mathbb{R}^D$ and the strictly
lower triangular matrix $\mathbf{L} \in \mathbb{R}^{D \times D}$, respectively.
Note that `l[k]` does not include the elements of $\mathbf{L}$ that are
guaranteed to be zero due to it being strictly lower triangular; that is,
`l[k][0]` is empty, `l[k][1]` has one element, `l[k][2]` has two elements, and
so on. From these, we construct the lower triangular matrix

$$
Q(\boldsymbol{q}, \mathbf{L}) = \begin{bmatrix}
  \exp(q_1) & 0 & \cdots & 0 \\
  l_{2,1} & \exp(q_2) & \cdots & 0 \\
  \vdots & \vdots & \ddots & \vdots \\
  l_{D,1} & l_{D,2} & \cdots & \exp(q_D)
\end{bmatrix} \in \mathbb{R}^{D \times D}
$$

by exponentiating each value of $\boldsymbol{q}$ to form the diagonal and then
summing with $\mathbf{L}$. Then we use this to compute the inverse of the
positive-definite covariance matrix as
$\mathbf{\Sigma}_k^{-1} = Q(\boldsymbol{q}, \mathbf{L})Q(\boldsymbol{q}, \mathbf{L})^\top \in \mathbb{R}^{D \times D}$.
Conceptually, the concatenation of these covariance matrices could be considered
to form a tensor $\mathsf{\Sigma} \in \mathbb{R}^{K \times D \times D}$.

The field `alpha` encodes the vector $\boldsymbol{\alpha} \in \mathbb{R}^K$,
which parameterizes the weights $\boldsymbol{w} \in \mathbb{R}^K$ as

$$w_k = \frac{\exp(\alpha_k)}{\sum_{k'=1}^K \exp(\alpha_{k'})}$$

which ensures that $\sum_{k=1}^K w_k = 1$.

From these values, we define the GMM likelihood function

$$p(\mathbf{X}; \boldsymbol{w}, \mathbf{M}, \mathsf{\Sigma}) = \prod_{i=1}^N \sum_{k=1}^K w_k |2\pi\mathbf{\Sigma}_k|^{-\frac{1}{2}} \exp\bigg(-\frac{1}{2}(\boldsymbol{x}_i - \boldsymbol{\mu}_k)^\top \Sigma_k^{-1}(\boldsymbol{x}_i - \boldsymbol{\mu}_k)\bigg).$$

The fields `m` and `gamma` encode $m \in \mathbb{Z}$ and $\gamma \in \mathbb{R}$
respectively, parameterizing an Identity-Wishart prior over the covariances as

$$p(\mathsf{\Sigma}) = \prod_{k=1}^K C(D, m) |\mathbf{\Sigma}_k|^m \exp\bigg(-\frac{1}{2}\,\text{trace}(\mathbf{\Sigma}_k)\bigg)$$

where $C(D, m) = \text{TODO}$. Note that the original ADBench paper does
explicitly define $C$, and does not mention $\gamma$ at all; these are
reconstructed by looking at their implementation.

The GMM `objective` function is then defined as the negative log posterior

$$L(\boldsymbol{w}, \mathbf{M}, \mathsf{\Sigma}; \mathbf{X}) = -\log\big(p(\mathbf{X}; \boldsymbol{w}, \mathbf{M}, \mathsf{\Sigma})p(\mathsf{\Sigma})\big)$$

and the `jacobian` function computes $\nabla L$ with respect to the four `Input`
fields encoding all the `Independent` variables $\boldsymbol{\alpha}$,
$\mathbf{M}$, $\mathbf{Q}$, and $\mathsf{L}$.

## Implementation

To actually _compute_ `objective`, it is typical to first perform further
algebraic simplifications, since the definitions of the likelihood and prior
include determinants and matrix inversions that would be expensive to compute
naively. Specifically,

$$
\begin{aligned}
\log p(\mathbf{X}; \boldsymbol{w}, \mathbf{M}, \mathsf{\Sigma})
&= \frac{ND}{2} \log 2\pi \\
&- \sum_{i=1}^N \text{logsumexp}\Bigg(\bigg[\alpha_k + \sum_{k=1}^K \boldsymbol{q}_k - \frac{1}{2}\|Q(\boldsymbol{q}_k, \boldsymbol{l}_k)(\boldsymbol{x}_i - \boldsymbol{\mu}_k)\|^2\bigg]_{k=1}^K\Bigg) \\
&+ N\,\text{logsumexp}\Big([\alpha_k]_{k=1}^K\Big)
\end{aligned}
$$

where the notation $[\alpha_k]_{k=1}^K$ means to construct a $K$-dimensional
vector in which the element at index $k$ is $\alpha_k$. Similarly, we can
simplify

$$\log p(\mathsf{\Sigma}) = -\frac{1}{2} \sum_{k=1}^K \gamma^2\big(\|\exp(\boldsymbol{q}_k)\|^2 + \|\boldsymbol{L}_k\|^2\big) + m \sum_{k=1}^K \boldsymbol{q}_k + K\bigg(c - \log \Gamma_p\Big(\frac{n}{2}\Big)\bigg)$$

where $c = ND \log \frac{\gamma}{\sqrt{2}}$. Finally, we can compute the overall
negative log posterior as

$$L(\boldsymbol{w}, \mathbf{M}, \mathsf{\Sigma}; \mathbf{X}) = -\log p(\mathbf{X}; \boldsymbol{w}, \mathbf{M}, \mathsf{\Sigma}) - \log p(\mathsf{\Sigma}).$$

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
[cpp]: /cpp/gradbench/evals/gmm.hpp
[futhark]: /tool/futhark/gmm.fut
[pytorch]: /python/gradbench/gradbench/tools/pytorch/gmm_objective.py
[`llsq`]: /evals/llsq
[`lstm`]: /evals/lstm
[`ode`]: /evals/ode
