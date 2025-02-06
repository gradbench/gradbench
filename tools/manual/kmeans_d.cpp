#include "kmeans_d.h"

#include <gradbench/kmeans.h>
#include <vector>
#include <cmath>

void kmeans_objective_d(int n, int k, int d, double const *points,
                        double const *centroids,
                        double *dir) {
  std::vector<int> cluster_sizes(k);
  std::vector<double> cluster_sums(k*d);

  for (int i = 0; i < n; i++) {
    double const *a = &points[i*d];
    double closest = INFINITY;
    int closest_j = -1;
    for (int j = 0; j < k; j++) {
      double const *b = &centroids[j*d];
      double dist = euclid_dist_2(d, a, b);
      if (dist < closest) {
        closest = dist;
        closest_j = j;
      }
    }
    cluster_sizes[closest_j]++;
    for (int l = 0; l < d; l++) {
      cluster_sums[closest_j*d+l] += a[l];
    }
  }

  for (int j = 0; j < k; j++) {
    double *cluster_sum = &cluster_sums.data()[j*d];
    int cluster_size = cluster_sizes[j];
    double *centroid_dir = &dir[j*d];
    double const *centroid = &centroids[j*d];
    for (int l = 0; l < d; l++) {
      centroid_dir[l] = -(cluster_sum[l]/cluster_size - centroid[l]);
    }
  }
}
