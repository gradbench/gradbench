#include "gradbench/KMeansData.h"
#include "gradbench/kmeans.h"
#include "gradbench/main.h"
#include "kmeans_d.h"

class Cost : public Function<kmeans::Input, kmeans::CostOutput> {
public:
  Cost(kmeans::Input& input) : Function(input) {}
  void compute(kmeans::CostOutput& output) {
    output = kmeans_objective(_input.n, _input.k, _input.d,
                              _input.points.data(),
                              _input.centroids.data());
  }
};

class Dir : public Function<kmeans::Input, kmeans::DirOutput> {
public:
  Dir(kmeans::Input& input) : Function(input) {}
  void compute(kmeans::DirOutput& output) {
    output.k = _input.k;
    output.d = _input.d;
    output.dir.resize(_input.k * _input.d);
    kmeans_objective_d(_input.n, _input.k, _input.d,
                       _input.points.data(),
                       _input.centroids.data(),
                       output.dir.data());
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv, {
      {"cost", function_main<Cost>},
      {"dir", function_main<Dir>}
    });
}
