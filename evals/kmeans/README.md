# K-means clustering

K-means clustering is an algorithm for partitioning *n*
*d*-dimensional observations into *k* clusters, such that we minimise
the sum of distances from each point to the centroid of its cluster.
This can be done using Newton's Method, which requires computing the
Hessian of an appropriate cost function. The Hessian also happens to
be sparse - it only has nonzero elements along the diagonal.
