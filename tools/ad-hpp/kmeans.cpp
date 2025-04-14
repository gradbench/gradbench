#include <algorithm>
#include <vector>

#include "gradbench/evals/kmeans.hpp"
#include "gradbench/main.hpp"

#include "ad.hpp"

using tangent_t = ad::tangent_t<double>;
using adjoint_hess_t = ad::adjoint_t<tangent_t>;
using adjoint_hess = ad::adjoint<tangent_t>;
using tape_t = adjoint_hess::tape_t;

class Dir : public Function<kmeans::Input, kmeans::DirOutput> {
  tape_t *_tape;

public:
  Dir(kmeans::Input &input) : Function(input) { _tape = tape_t::create(); }
  ~Dir() { tape_t::remove(_tape); }

  void compute(kmeans::DirOutput &output) {
    output.k = _input.k;
    output.d = _input.d;
    output.dir.resize(_input.k * _input.d);

    std::vector<adjoint_hess_t> points_active(_input.points.size());
    std::vector<adjoint_hess_t> centroids_active(_input.centroids.size());

    for (size_t i = 0; i < centroids_active.size(); i++) {
      ad::passive_value(centroids_active[i]) = _input.centroids[i];
      // being extra careful (for didactic reasons?)
      ad::derivative(ad::value(centroids_active[i])) = 0;
      ad::value(ad::derivative(centroids_active[i])) = 0;
      ad::derivative(ad::derivative(centroids_active[i])) = 0;
    }

    std::for_each(
        centroids_active.begin(), centroids_active.end(),
        [](adjoint_hess_t &var) -> void { _tape->register_variable(var); });
    tape_t::position_t tape_pos = _tape->get_position();

    adjoint_hess_t active_err;
    kmeans::objective(_input.n, _input.k, _input.d, points_active.data(),
                      centroids_active.data(), &active_err);
  }
};
