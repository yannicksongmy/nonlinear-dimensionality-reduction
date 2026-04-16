# Principal Component Analysis (PCA)
PCA is a technique that is widely used for applications such as dimensionalitu reduction. It was first introduced by [Karl Pearson](https://www.tandfonline.com/doi/abs/10.1080/14786440109462720) in 1901, as the linear projection that minimizes the mean squared distance between the data points and their projections.
However, the well-known formulation was given by [Harold Hotteling](https://psycnet.apa.org/doiLanding?doi=10.1037%2Fh0070888) in 1933.

## The magic behin PCA
PCA finds a new orthonormal basis that best re-expresses the data in a _d_-dimensional space, where _d_ is a linear combination of the _D_ original dimensions.

The axes of that new basis need to be **linearly independent** to decorrelate the data. 

The main goal of that basis is to **maximize the variance** of the data along its axes, which are called **principal components**.

## Dimensionality reduction with PCA
To reduce the dimensionality of the data, PCA does not consider the axe where the variability of the data is not significant.

## Steps for implementing PCA

**Step 1: Normalize the data**
The first step is to center the data by calculating the mean $\mu$ of the data. Then, for each data point $X_i$, substract the mean $\mu$ to center the data around the origine as follows:

$X_i^' = X_i - \mu$

You can also need to reduce the data if they have not the same scale. Thus, the first step become:

$X_i^' = (X_i - \mu)/\sigma$
where $\sigma$ is variance of the data.

**Step 2: Compute the covariance matrix of the normalized data**
The covariance matrix $C_X$ of the normalized data is computed as follows:

$ C_X = \frac{1}{N} \sum_{i=1}^N X_i^'(X_i)^T$
where $N$ is the number of data points.

**Step 3: Perform the eigenvector decomposition**
The eigenvector decomposition of the covariance matrix $C_X$ is perform to obtain the principal components as follows:

$C_X = USU^T$
where $U$ contains the eigenvectors _(principal directions)_, and $S$ is a diagonal matrix that contains the eigenvalues _(explained variance)_.

**Step 4: Data projection**
The last step consists of projecting the data $X_i^'$ onto the principal directions as follows:

$Y_i = V^TX_i^'$
where $Y_i$ represents the data in the new reduced space.

## Experiments
We implement PCA step by step in this directory, and compare the results with the PCA function from sklearn. The obtained results are the same!

## Note
To well understand the mathematics behind PCA, read the **main.pdf** file in this directory.