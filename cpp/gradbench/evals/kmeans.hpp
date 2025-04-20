#pragma once

#include "gradbench/main.hpp"
#include "json.hpp"
#include <cmath>
#include <vector>

namespace kmeans {
template <typename A, typename B>
B euclid_dist_2(int d, A const* a, B const* b) {
  B dist = 0;
  for (int i = 0; i < d; i++) {
    dist += (a[i] - b[i]) * (a[i] - b[i]);
  }
  return dist;
}

template <typename T>
void objective(int n, int k, int d, double const* __restrict__ points,
               T const* __restrict__ centroids, T* __restrict__ err) {
  T cost = 0;
  for (int i = 0; i < n; i++) {
    double const* a       = &points[i * d];
    T             closest = INFINITY;
    for (int j = 0; j < k; j++) {
      T const* b = &centroids[j * d];
      closest    = std::min(closest, euclid_dist_2(d, a, b));
    }
    cost += closest;
  }
  *err = cost;
}

struct Input {
  int                 n, k, d;
  std::vector<double> points;     // n*d matrix
  std::vector<double> centroids;  // k*d matrix
};

typedef double CostOutput;

struct DirOutput {
  int                 k, d;
  std::vector<double> dir;  // k*d matrix
};

using json = nlohmann::json;

void from_json(const json& j, Input& p) {
  auto points    = j.at("points").get<std::vector<std::vector<double>>>();
  auto centroids = j.at("centroids").get<std::vector<std::vector<double>>>();

  int n = points.size();
  int d = points[0].size();
  int k = centroids.size();

  p.n = n;
  p.k = k;
  p.d = d;

  for (int i = 0; i < n; i++) {
    p.points.insert(p.points.end(), points[i].begin(), points[i].end());
  }

  for (int i = 0; i < k; i++) {
    p.centroids.insert(p.centroids.end(), centroids[i].begin(),
                       centroids[i].end());
  }
}

void to_json(nlohmann::json& j, const DirOutput& p) {
  std::vector<std::vector<double>> out(p.k);
  for (int i = 0; i < p.k; i++) {
    out[i].resize(p.d);
    for (int j = 0; j < p.d; j++) {
      out[i][j] = p.dir[i * p.d + j];
    }
  }
  j = out;
}

class Cost : public Function<kmeans::Input, kmeans::CostOutput> {
public:
  Cost(kmeans::Input& input) : Function(input) {}
  void compute(kmeans::CostOutput& output) {
    objective(_input.n, _input.k, _input.d, _input.points.data(),
              _input.centroids.data(), &output);
  }
};
}  // namespace kmeans
