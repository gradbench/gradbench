# ADBench Functions

Based on their python implementation

## Gaussian Mixture Model Fitting (GMM)

[Link to IO file](https://github.com/microsoft/ADBench/blob/master/src/python/shared/GMMData.py) and [data](https://github.com/microsoft/ADBench/tree/master/data/gmm)

[Link to GMM Data Generator](https://github.com/microsoft/ADBench/blob/master/data/gmm/gmm-data-gen.py)

### Inputs

1. alphas
2. means
3. icf
4. x
5. Wishart

These are all conditional on a D and K. D ranges from $2^0$ to $2^7$ and represents the dimension of the data points and means. K is the number of mixture components (clusters) where K $\in$ [5,10,25,50,100]. Additionally GMM can be run with 1k, 10k, or 2.5 million data points.

### Outputs

1. Objective Value
2. Gradient Array

## Bundle Adjustment (BA)

[Link to IO file](https://github.com/microsoft/ADBench/blob/master/src/python/shared/BAData.py) and [data](https://github.com/microsoft/ADBench/tree/master/data/ba)

### Inputs

1. Cams
2. X
3. W
4. obs
5. feats

### Outputs

1. Reproj Errror
2. W Error
3. Sparse Jacobian

## Hand Track (HT)

[Link to IO file](https://github.com/microsoft/ADBench/blob/master/src/python/shared/HandData.py) and [data](https://github.com/microsoft/ADBench/tree/master/data/hand)

### Inputs

1. Theta
2. Data
3. Us

### Outputs

1. Objective Array
2. Jacobian Array
