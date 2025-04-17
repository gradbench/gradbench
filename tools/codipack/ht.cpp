#include "gradbench/main.hpp"
#include "gradbench/evals/ht.hpp"
#include "codi_impl.hpp"

class Jacobian : public Function<ht::Input, ht::JacOutput>, CoDiForwardRunnerVec<4> {
  using Real = typename CoDiForwardRunnerVec<4>::Real;

  bool _complicated = false;
  std::vector<double> _objective;

  std::vector<Real> ad_err;
  std::vector<Real> ad_us;
  std::vector<Real> ad_theta;

public:
  Jacobian(ht::Input& input) :
    Function(input),
    _complicated(input.us.size() != 0),
    _objective(3 * input.data.correspondences.size()),
    ad_err(_objective.size()),
    ad_us(_objective.size()),
    ad_theta(input.theta.size())
  {
    std::copy(input.theta.begin(), input.theta.end(), ad_theta.begin());
    std::copy(input.us.begin(), input.us.end(), ad_us.begin());
  }

  void compute(ht::JacOutput& output) {
    int err_size = 3 * _input.data.correspondences.size();
    int ncols = (_complicated ? 2 : 0) + _input.theta.size();
    output.jacobian_ncols = ncols;
    output.jacobian_nrows = err_size;
    output.jacobian.resize(err_size * ncols);

    if (!_complicated) {
      compute_ht_simple_J(output.jacobian);
    } else {
      compute_ht_complicated_J(output.jacobian);
    }
  }

  void compute_ht_simple_J(std::vector<double>& J) {

    int runs = codiGetRuns(ad_theta.size());

    for(int r = 0; r < runs; r += 1) {
      int offset = r * dim;
      int cur_dim_size = std::min(dim, (int)ad_theta.size() - offset);

      for(int d = 0; d < cur_dim_size; d += 1) {
        codiSetGradient(ad_theta[offset + d], d, 1.0);
      }

      ht::objective<Real>(ad_theta.data(), &_input.data, ad_err.data());
      int Jrows = 3* _input.data.correspondences.size();

      for(int d = 0; d < cur_dim_size; d += 1) {
        codiSetGradient(ad_theta[offset + d], d, 0.0);

        for (size_t j = 0; j < ad_err.size(); j++) {
          J[(offset + d)*Jrows+j] = codiGetGradient(ad_err[j], d);
        }
      }
    }
  }

  void compute_ht_complicated_J(std::vector<double>& J) {

    int total_size = 2+ad_theta.size();
    int runs = codiGetRuns(total_size);

    for(int r = 0; r < runs; r += 1) {
      int offset = r * dim;
      int cur_dim_size = std::min(dim, (int)total_size - offset);

      for(int d = 0; d < cur_dim_size; d += 1) {
        int i = offset + d;

        if (i >= 2) {
          codiSetGradient(ad_theta[i-2], d, 1.0);
        } else {
          for (size_t j = 0; j < ad_us.size(); j++) {
            codiSetGradient(ad_us[j], d, (1+i+j) % 2);
          }
        }
      }

      ht::objective(ad_theta.data(),
                    ad_us.data(),
                    &_input.data,
                    ad_err.data());
      int Jrows = 3*_input.data.correspondences.size();

      for(int d = 0; d < cur_dim_size; d += 1) {
        int i = offset + d;

        for (size_t j = 0; j < ad_err.size(); j++) {
          J[i*Jrows+j] = codiGetGradient(ad_err[j], d);
        }
        if (i >= 2) {
          codiSetGradient(ad_theta[i-2], d, 0.0);
        } else {
          for (size_t j = 0; j < ad_us.size(); j++) {
            codiSetGradient(ad_us[j], d, 0.0);
          }
        }
      }
    }
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv, {
      {"objective", function_main<ht::Objective>},
      {"jacobian", function_main<Jacobian>}
    });
}
