# Gaussian Mixture Model Fitting (GMM)

Information on the GMM equation from Microsoft's [ADBench](https://github.com/microsoft/ADBench/tree/38cb7931303a830c3700ca36ba9520868327ac87) based on their python implementation

## Gaussian Mixture Model Fitting (GMM)

Link to [I/O file](https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/python/shared/GMMData.py), [data folder](https://github.com/microsoft/ADBench/tree/38cb7931303a830c3700ca36ba9520868327ac87/data/gmm), and [GMM Data Generator](https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/data/gmm/gmm-data-gen.py)

### Generation

To generate files with the below inputs, 3 values are used: D, K, and N. D ranges from $2^1$ to $2^7$ and represents the dimension of the data points and means. K is the number of mixture components (clusters) where K $\in [5,10,25,50,100,200]$. Additionally, GMM can be run with $1000$ or $10000$ data points, where N represents this value. These values/ranges are iterated over to create various datasets.

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
2. Gradient ($G$) of Log-Likelihood: How it will change given changes to alphas, means, and ICF
   $$G \in \mathbb{R}^{K + (K \times D) + (K \times (D + \frac{D(D-1)}{2}))}$$

> **Example**
>
> If $D = 2$ and $K = 5$, $G \in \mathbb{R}^{30}$ meaning the function will return an array of length 30.
