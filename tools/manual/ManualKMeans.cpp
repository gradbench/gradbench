#include "ManualKMeans.h"
#include "adbench/shared/kmeans.h"
#include "kmeans_d.h"

#include <iostream>

ManualKMeans::ManualKMeans(KMeansInput& input) : ITest(input) {
  _output = {
    input.n, input.k, input.d,
    0,
    std::vector<double>(input.points.size())
  };
}

void ManualKMeans::calculate_objective() {
  _output.cost = kmeans_objective(_input.n, _input.k, _input.d,
                                  _input.points.data(),
                                  _input.centroids.data());
}

void ManualKMeans::calculate_jacobian() {
  kmeans_objective_d(_input.n, _input.k, _input.d,
                     _input.points.data(),
                     _input.centroids.data(),
                     _output.dir.data());
}
