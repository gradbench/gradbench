#include "gradbench/evals/ht.hpp"
#include "gradbench/main.hpp"
#include <codi.hpp>

using Real = codi::RealForward;

void compute_ht_complicated_J(const std::vector<double>& theta,
                              const std::vector<double>& us,
                              const ht::DataLightMatrix& data,
                              std::vector<double>&       err,
                              std::vector<double>&       J) {
  std::vector<Real> ad_err(err.size());
  std::vector<Real> ad_us(err.size());
  std::vector<Real> ad_theta(theta.size());
  std::copy(theta.begin(), theta.end(), ad_theta.begin());
  std::copy(us.begin(), us.end(), ad_us.begin());

  for (size_t i = 0; i < 2 + theta.size(); i++) {
    if (i >= 2) {
      ad_theta[i - 2].setGradient(1);
    } else {
      for (size_t j = 0; j < ad_us.size(); j++) {
        ad_us[j].setGradient((1 + i + j) % 2);
      }
    }
    ht::objective(ad_theta.data(), ad_us.data(), &data, ad_err.data());
    int Jrows = 3 * data.correspondences.size();
    for (size_t j = 0; j < ad_err.size(); j++) {
      J[i * Jrows + j] = ad_err[j].gradient();
    }
    if (i >= 2) {
      ad_theta[i - 2].setGradient(0);
    } else {
      for (size_t j = 0; j < ad_us.size(); j++) {
        ad_us[j].setGradient(0);
      }
    }
  }
}

void compute_ht_simple_J(const std::vector<double>& theta,
                         const ht::DataLightMatrix& data,
                         std::vector<double>& err, std::vector<double>& J) {
  std::vector<Real> ad_err(err.size());
  std::vector<Real> ad_theta(theta.size());
  std::copy(theta.begin(), theta.end(), ad_theta.begin());

  for (size_t i = 0; i < theta.size(); i++) {
    ad_theta[i].setGradient(1);
    ht::objective<Real>(ad_theta.data(), &data, ad_err.data());
    ad_theta[i].setGradient(0);
    int Jrows = 3 * data.correspondences.size();
    for (size_t j = 0; j < ad_err.size(); j++) {
      J[i * Jrows + j] = ad_err[j].gradient();
    }
  }
}

class Jacobian : public Function<ht::Input, ht::JacOutput> {
  bool                _complicated = false;
  std::vector<double> _objective;

public:
  Jacobian(ht::Input& input)
      : Function(input), _complicated(input.us.size() != 0),
        _objective(3 * input.data.correspondences.size()) {}

  void compute(ht::JacOutput& output) {
    int err_size          = 3 * _input.data.correspondences.size();
    int ncols             = (_complicated ? 2 : 0) + _input.theta.size();
    output.jacobian_ncols = ncols;
    output.jacobian_nrows = err_size;
    output.jacobian.resize(err_size * ncols);

    if (!_complicated) {
      compute_ht_simple_J(_input.theta, _input.data, _objective,
                          output.jacobian);
    } else {
      compute_ht_complicated_J(_input.theta, _input.us, _input.data, _objective,
                               output.jacobian);
    }
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv,
                      {{"objective", function_main<ht::Objective>},
                       {"jacobian", function_main<Jacobian>}});
}
