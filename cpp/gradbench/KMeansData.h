#pragma once

#include <vector>
#include "json.hpp"

namespace kmeans {
  struct Input {
    int n, k, d;
    std::vector<double> points; // n*d matrix
    std::vector<double> centroids; // k*d matrix
  };

  typedef double CostOutput;

  struct DirOutput {
    int k, d;
    std::vector<double> dir; // k*d matrix
  };

  using json = nlohmann::json;

  void from_json(const json& j, Input& p) {
    auto points = j.at("points").get<std::vector<std::vector<double>>>();
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
      p.centroids.insert(p.centroids.end(), centroids[i].begin(), centroids[i].end());
    }
  }

  void to_json(nlohmann::json& j, const DirOutput& p) {
    std::vector<std::vector<double>> out(p.k);
    for (int i = 0; i < p.k; i++) {
      out[i].resize(p.d);
      for (int j = 0; j < p.d; j++) {
        out[i][j] = p.dir[i*p.d+j];
      }
    }
    j = out;
  }
}
