#include <algorithm>
#include <vector>

#include "ad.hpp"
#include "gradbench/evals/kmeans.hpp"
#include "gradbench/main.hpp"

using tangent_t      = ad::tangent_t<double>;
using adjoint_hess_t = ad::adjoint_t<tangent_t>;
using adjoint_hess   = ad::adjoint<tangent_t>;
using tape_t         = adjoint_hess::tape_t;

class Dir : public Function<kmeans::Input, kmeans::DirOutput> {
  tape_t* _tape;

public:
  /**
   * @brief
   *
   */
  Dir(kmeans::Input& input) : Function(input) { _tape = tape_t::create(); }
  ~Dir() { tape_t::remove(_tape); }

  void dAxb(std::vector<double> const& A_diag, std::vector<double> const& b,
            double* out) {
    assert(A_diag.size() == b.size() && "Check diagonal matmul size.");
  }

  void compute(kmeans::DirOutput& output) {
    output.k = _input.k;
    output.d = _input.d;
    output.dir.resize(_input.k * _input.d);

    std::vector<adjoint_hess_t> centroids_active(_input.centroids.size());
    for (size_t i = 0; i < centroids_active.size(); i++) {
      // being extra careful (for didactic reasons?)
      ad::passive_value(centroids_active[i])              = _input.centroids[i];
      ad::derivative(ad::value(centroids_active[i]))      = 0;
      ad::value(ad::derivative(centroids_active[i]))      = 0;
      ad::derivative(ad::derivative(centroids_active[i])) = 0;
      _tape->register_variable(centroids_active[i]);
    }
    tape_t::position_t tape_pos = _tape->get_position();

    std::vector<double> dfdci(_input.k * _input.d);
    std::vector<double> d2fdcii2(_input.k * _input.d);
    adjoint_hess_t      active_err;
    for (size_t hrow = 0; hrow < centroids_active.size(); hrow++) {
      // input perturbation
      ad::derivative(ad::value(centroids_active[hrow])) = 1;
      kmeans::objective(_input.n, _input.k, _input.d, _input.points.data(),
                        centroids_active.data(), &active_err);
      // output sensitivity
      ad::value(ad::derivative(active_err)) = 1;
      _tape->interpret_adjoint_and_reset_to(tape_pos);
      // only copy gradient on first run
      if (hrow == 0) {
        std::transform(centroids_active.begin(), centroids_active.end(),
                       dfdci.begin(), [&](auto const& v) -> double {
                         return ad::derivative(ad::value(v));
                       });
      }
      // copy hessian and reset accumulated values
      d2fdcii2[hrow] = ad::derivative(ad::derivative(centroids_active[hrow]));
      ad::derivative(ad::derivative(centroids_active[hrow])) = 0;
      ad::value(ad::derivative(centroids_active[hrow]))      = 0;
      ad::derivative(ad::value(centroids_active[hrow]))      = 0;
    }
  }
};
