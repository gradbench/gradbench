#include "gradbench/kmeans.hpp"
#include "gradbench/main.hpp"
#include "enzyme.h"

void kmeans_objective_d(int n, int k, int d,
                        double const *points,
                        double const *centroids, double *centroids_J) {
  double err;
  double err_seed = 1;
  __enzyme_autodiff(kmeans::objective<double>,
                    enzyme_const, n,
                    enzyme_const, k,
                    enzyme_const, d,

                    enzyme_const, points,
                    enzyme_dup, centroids, centroids_J,

                    enzyme_dupnoneed, &err, &err_seed);
}

void kmeans_objective_dd(int n, int k, int d,
                         double const *points,
                         double const *centroids,
                         double const* centroids_seed,
                         double *centroids_J,
                         double *centroids_H) {
  __enzyme_fwddiff(kmeans_objective_d,
                   enzyme_const, n,
                   enzyme_const, k,
                   enzyme_const, d,

                   enzyme_const, points,
                   enzyme_dup, centroids, centroids_seed,
                   enzyme_dup, centroids_J, centroids_H);
}

class Dir : public Function<kmeans::Input, kmeans::DirOutput> {
private:
  std::vector<double> _centroids_seed;
  std::vector<double> _centroids_J;
  std::vector<double> _centroids_H;

public:

  Dir(kmeans::Input& input) :
    Function(input),
    _centroids_seed(input.k*input.d),
    _centroids_J(input.k * input.d),
    _centroids_H(input.k * input.d) {
    for (int i = 0; i < input.k*input.d; i++) {
      _centroids_seed[i] = 1;
    }
  }

  void compute(kmeans::DirOutput& output) {
    output.k = _input.k;
    output.d = _input.d;
    output.dir.resize(_input.k * _input.d);

    kmeans_objective_dd(_input.n, _input.k, _input.d,
                        _input.points.data(),
                        _input.centroids.data(),
                        _centroids_seed.data(),
                        _centroids_J.data(),
                        _centroids_H.data());

    for (int i = 0; i < _input.k * _input.d; i++){
      output.dir[i] = _centroids_J[i] / _centroids_H[i];
    }
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv, {
      {"cost", function_main<kmeans::Cost>},
      {"dir", function_main<Dir>}
    });
}
