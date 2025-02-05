#pragma once

#include <vector>

struct KMeansInput {
  int n, k, d;
  std::vector<double> points; // n*d matrix
  std::vector<double> centroids; // k*d matrix
};

struct KMeansOutput {
  int n, k, d;
  double cost;
  std::vector<double> dir; // k*d matrix
};
