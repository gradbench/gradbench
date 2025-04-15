#include "gradbench/evals/kmeans.hpp"
#include "gradbench/main.hpp"
#include <codi.hpp>

using t1s = codi::RealForwardGen<double>;
using r2s = codi::RealReverseGen<t1s>;

class Dir : public Function<kmeans::Input, kmeans::DirOutput> {
public:

  Dir(kmeans::Input& input) : Function(input) {}

  void compute(kmeans::DirOutput& output) {
    using DH = codi::DerivativeAccess<r2s>;
    output.k = _input.k;
    output.d = _input.d;
    output.dir.resize(_input.k * _input.d);

    std::vector<r2s> ad_centroids(_input.centroids.size());

    std::copy(_input.centroids.begin(), _input.centroids.end(),
              ad_centroids.begin());

    r2s::Tape& tape = r2s::getTape();
    tape.setActive();

    for (auto &v : ad_centroids) {
      DH::setAllDerivativesForward(v, 1, 1.0);
      tape.registerInput(v);
    }

    r2s err;

    kmeans::objective<r2s>(_input.n, _input.k, _input.d,
                           _input.points.data(),
                           ad_centroids.data(),
                           &err);

    tape.registerOutput(err);

    DH::setAllDerivativesReverse(err, 1, 1.0);

    tape.setPassive();
    tape.evaluate();

    for (int i = 0; i < _input.k * _input.d; i++){
      output.dir[i] =
        DH::derivative(ad_centroids[i], 1, 1)/
        DH::derivative(ad_centroids[i], 2, 0);
    }
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv, {
      {"cost", function_main<kmeans::Cost>},
      {"dir", function_main<Dir>}
    });
}
