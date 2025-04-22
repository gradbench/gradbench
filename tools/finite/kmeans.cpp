#include "gradbench/evals/kmeans.hpp"
#include "finite.h"
#include "gradbench/main.hpp"

class Dir : public Function<kmeans::Input, kmeans::DirOutput> {
  FiniteDifferencesEngine<double> _engine;

public:
  Dir(kmeans::Input& input) : Function(input), _engine(1) {}
  void compute(kmeans::DirOutput& output) {
    output.k = _input.k;
    output.d = _input.d;
    output.dir.resize(_input.k * _input.d);
    std::vector<double> J(output.dir.size());
    std::vector<double> H(output.dir.size());

    _engine.finite_differences(
        1,
        [&](double* centroids_in, double* err) {
          kmeans::objective(_input.n, _input.k, _input.d, _input.points.data(),
                            centroids_in, err);
        },
        _input.centroids.data(), _input.centroids.size(), 1, J.data());

    _engine.finite_differences(
        2,
        [&](double* centroids_in, double* err) {
          kmeans::objective(_input.n, _input.k, _input.d, _input.points.data(),
                            centroids_in, err);
        },
        _input.centroids.data(), _input.centroids.size(), 1, H.data());

    for (size_t i = 0; i < J.size(); i++) {
      output.dir[i] = J[i] / H[i];
    }
  }
};

int main(int argc, char* argv[]) {
  return generic_main(
      argc, argv,
      {{"cost", function_main<kmeans::Cost>}, {"dir", function_main<Dir>}});
}
