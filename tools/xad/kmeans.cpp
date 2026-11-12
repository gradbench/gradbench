#include "gradbench/evals/kmeans.hpp"
#include "gradbench/main.hpp"
#include <XAD/XAD.hpp>
#include <vector>

// Second order via forward-over-adjoint: one recording gives the
// gradient J and the Hessian-vector product H*v. The Hessian of the
// k-means cost is diagonal, so seeding the tangent with v = (1, ..., 1)
// yields diag(H) exactly. CoDiPack uses the same diagonal trick.
class Dir : public Function<kmeans::Input, kmeans::DirOutput> {
  using mode = xad::fwd_adj<double>;
  using AD   = mode::active_type;

  mode::tape_type _tape;

public:
  Dir(kmeans::Input& input) : Function(input) {}

  void compute(kmeans::DirOutput& output) {
    output.k = _input.k;
    output.d = _input.d;
    output.dir.resize(_input.k * _input.d);

    _tape.clearAll();
    std::vector<AD> centroids(_input.centroids.begin(), _input.centroids.end());
    _tape.registerInputs(centroids);
    for (auto& c : centroids) {
      xad::derivative(xad::value(c)) = 1.0;
    }
    _tape.newRecording();

    AD err;
    kmeans::objective<AD>(_input.n, _input.k, _input.d, _input.points.data(),
                          centroids.data(), &err);

    _tape.registerOutput(err);
    xad::value(xad::derivative(err)) = 1.0;
    _tape.computeAdjoints();

    for (size_t i = 0; i < centroids.size(); i++) {
      output.dir[i] = xad::value(xad::derivative(centroids[i])) /
                      xad::derivative(xad::derivative(centroids[i]));
    }
  }
};

int main(int argc, char* argv[]) {
  return generic_main(
      argc, argv,
      {{"cost", function_main<kmeans::Cost>}, {"dir", function_main<Dir>}});
}
