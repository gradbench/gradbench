#include "gradbench/KMeansData.h"
#include "gradbench/kmeans.h"
#include "gradbench/main.h"
#include "enzyme.h"

class Cost : public Function<kmeans::Input, kmeans::CostOutput> {
public:
  Cost(kmeans::Input& input) : Function(input) {}
  void compute(kmeans::CostOutput& output) {
    kmeans_objective(_input.n, _input.k, _input.d,
                     _input.points.data(),
                     _input.centroids.data(),
                     &output);
  }
};

void kmeans_objective_d(int n, int k, int d,
                        double const *points,
                        double const *centroids, double *d_centroids) {
  double err;
  double err_seed = 1;
  __enzyme_autodiff(kmeans_objective<double>,
                    enzyme_const, n,
                    enzyme_const, k,
                    enzyme_const, d,

                    enzyme_const, points,
                    enzyme_dup, centroids, d_centroids,

                    enzyme_dupnoneed, &err, &err_seed);
}

void kmeans_objective_dd(int n, int k, int d,
                        double const *points,
                        double const *centroids,
                        double *d_centroids,
                        double *dd_centroids) {
  std::vector<double> centroids_seed(k*d);
  for (int i = 0; i < k*d; i++) {
    centroids_seed[i] = 1;
  }

  __enzyme_fwddiff(kmeans_objective_d,
                   enzyme_const, n,
                   enzyme_const, k,
                   enzyme_const, d,

                   enzyme_const, points,
                   enzyme_dup, centroids, centroids_seed.data(),
                   enzyme_dup, d_centroids, dd_centroids);
}

class Dir : public Function<kmeans::Input, kmeans::DirOutput> {
public:
  Dir(kmeans::Input& input) : Function(input) {}
  void compute(kmeans::DirOutput& output) {
    output.k = _input.k;
    output.d = _input.d;
    output.dir.resize(_input.k * _input.d);

    std::vector<double> d_centroids(_input.k * _input.d);
    std::vector<double> dd_centroids(_input.k * _input.d);

    kmeans_objective_dd(_input.n, _input.k, _input.d,
                        _input.points.data(),
                        _input.centroids.data(),
                        d_centroids.data(),
                        dd_centroids.data());

    for (int i = 0; i < _input.k * _input.d; i++){
      output.dir[i] = d_centroids[i] / dd_centroids[i];
    }
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv, {
      {"cost", function_main<Cost>},
      {"dir", function_main<Dir>}
    });
}
