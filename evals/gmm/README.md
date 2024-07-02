# Gaussian Mixture Model Fitting (GMM)

Information on functions from Microsoft's [ADBench](https://github.com/microsoft/ADBench/tree/38cb7931303a830c3700ca36ba9520868327ac87) based on their python implementation

## Gaussian Mixture Model Fitting (GMM)

Link to [I/O file](https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/python/shared/GMMData.py), [data folder](https://github.com/microsoft/ADBench/tree/38cb7931303a830c3700ca36ba9520868327ac87/data/gmm), and [GMM Data Generator](https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/data/gmm/gmm-data-gen.py)

### Generation

To generate files with the below inputs, 3 values are used: D, K, and N. D ranges from $2^0$ to $2^7$ and represents the dimension of the data points and means. K is the number of mixture components (clusters) where K $\in$ [5,10,25,50,100]. Additionally, GMM can be run with 1k, 10k, or 2.5 million data points, where N represents this value. These values/ranges are iterated over in the data generator.

### Inputs

One file is read in that then extracts the following inputs

1. Alphas ($\alpha$): Mixing components, weights
   $$
   \begin{align*}
   \alpha \in \R^K
   \end{align*}
   $$
2. Means ($M$): Expected centroid points, $\mu_k \in \R^D$
   $$
   \begin{align*}
   M \in \R^{K \times D}
   \end{align*}
   $$
3. Inverse Covariance Factor ($ICF$): Parameteres for the inverse covariance matrix (precision matrix)
   $$
   \begin{align*}
   ICF \in \R^{K \times (D + \frac{D(D-1)}{2})}
   \end{align*}
   $$
4. $X$: Data points being fitted
   $$
   \begin{align*}
   X \in \R^{D}
   \end{align*}
   $$
5. Wishart: Wishard distribution parameters to specify inital beliefs about scale and structure of precision matrcies stored in a tuple

### Outputs

1. Log-Likelihood Value: How well given parameteres fit the given data
2. Gradient ($G$) of Log-Likelihood: How it will change given changes to alphas, means, and ICF
   $$
   \begin{align*}
   G \in \R^{K + (K \times D) + (K \times (D + \frac{D(D-1)}{2}))}
   \end{align*}
   $$

> **Example**
>
> If $D = 2$ and $K = 5$, $G \in \R^{30}$ meaning the function will return an array of length 30.
