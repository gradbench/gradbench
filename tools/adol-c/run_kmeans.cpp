#include "gradbench/evals/kmeans.hpp"
#include "gradbench/main.hpp"

#include <adolc/adouble.h>
#include <adolc/drivers/drivers.h>
#include <adolc/taping.h>

static const int tapeTag = 1;

class Dir : public Function<kmeans::Input, kmeans::DirOutput> {
  std::vector<double> _H, _J, _tangent;
public:
  Dir(kmeans::Input& input)
    : Function(input),
      _H(input.k*input.d),
      _J(input.k*input.d),
      _tangent(input.centroids.size()) {
    std::vector<adouble> apoints(input.points.size());
    std::vector<adouble> acentroids(input.centroids.size());
    adouble aerr;

    trace_on(tapeTag);

    for (size_t i = 0; i < apoints.size(); i++) {
      apoints[i] = _input.points[i];
    }
    for (size_t i = 0; i < acentroids.size(); i++) {
      acentroids[i] <<= _input.centroids[i];
    }

    kmeans::objective<adouble>(_input.n, _input.k, _input.d,
                               apoints.data(), acentroids.data(),
                               &aerr);
    double err;
    aerr >>= err;

    trace_off();

    std::fill(_tangent.begin(), _tangent.end(), 1);
  }

  void compute(kmeans::DirOutput& output) {
    output.k = _input.k;
    output.d = _input.d;
    output.dir.resize(_input.k * _input.d);

    gradient(tapeTag, _input.centroids.size(),
             _input.centroids.data(), _J.data());

    hess_vec(tapeTag, _input.centroids.size(), _input.centroids.data(),
             _tangent.data(), _H.data());

    for (int i = 0; i < _input.k * _input.d; i++) {
      output.dir[i] = _J[i] / _H[i];
    }
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv, {
      {"cost", function_main<kmeans::Cost>},
      {"dir", function_main<Dir>}
    });
}
