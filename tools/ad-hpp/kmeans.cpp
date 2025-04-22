#include <vector>

#include "ad.hpp"
#include "gradbench/evals/kmeans.hpp"
#include "gradbench/main.hpp"

class Dir : public Function<kmeans::Input, kmeans::DirOutput> {
  using inner_ad_t     = ad::tangent_t<double>;
  using active_t       = ad::adjoint_t<inner_ad_t>;
  using adjoint        = ad::adjoint<inner_ad_t>;
  using tape_t         = adjoint::tape_t;
  using tape_options_t = adjoint::tape_options_t;

  tape_t* _tape;

public:
  Dir(kmeans::Input& input) : Function(input) {
    // 500 MiB???
    // long int       tape_size = 1024 * 1024 * 500;
    tape_options_t opts(AD_DEFAULT_TAPE_SIZE);
    _tape = tape_t::create(opts);
  }
  ~Dir() { tape_t::remove(_tape); }

  void compute(kmeans::DirOutput& output) {
    output.k = _input.k;
    output.d = _input.d;
    output.dir.resize(_input.k * _input.d);

    _tape->reset();
    std::vector<active_t> centroids_active(_input.centroids.size());
    for (size_t i = 0; i < centroids_active.size(); i++) {
      // set value
      ad::passive_value(centroids_active[i]) = _input.centroids[i];
      // clear {c_i}_(1)
      ad::value(ad::derivative(centroids_active[i])) = 0;
      // set {c_i}^(2) to extract diagonal hessian
      ad::derivative(ad::value(centroids_active[i])) = 1.0;
      // clear {c_i}_(1)^(2)
      ad::derivative(ad::derivative(centroids_active[i])) = 0;
      _tape->register_variable(centroids_active[i]);
    }

    active_t active_err;
    kmeans::objective(_input.n, _input.k, _input.d, _input.points.data(),
                      centroids_active.data(), &active_err);

    ad::value(ad::derivative(active_err)) = 1.0;
    _tape->interpret_adjoint();
    for (size_t i = 0; i < centroids_active.size(); i++) {
      // compute J_i * H^-1_{ii} and write to out.dir[i]
      output.dir[i] = ad::value(ad::derivative(centroids_active[i])) /
                      ad::derivative(ad::derivative(centroids_active[i]));
    }
  }
};

int main(int argc, char* argv[]) {
  return generic_main(
      argc, argv,
      {{"cost", function_main<kmeans::Cost>}, {"dir", function_main<Dir>}});
}
