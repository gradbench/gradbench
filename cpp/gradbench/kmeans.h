// Templated implementation of the kmeans cost function.

#pragma once

#include <iostream>
#include <cmath>

template<typename T>
T euclid_dist_2(int d, T const *a, T const *b) {
  T dist = 0;
  for (int i = 0; i < d; i++) {
    dist += (a[i]-b[i])*(a[i]-b[i]);
  }
  return dist;
}

template<typename T>
void kmeans_objective(int n, int k, int d,
                      T const *points, T const *centroids,
                      T* err) {
  T cost = 0;
  for (int i = 0; i < n; i++) {
    T const *a = &points[i*d];
    T closest = INFINITY;
    for (int j = 0; j < k; j++) {
      T const *b = &centroids[j*d];
      T dist = euclid_dist_2(d, a, b);
      if (dist < closest) {
        closest = dist;
      }
    }
    cost += closest;
  }
  *err = cost;
}
