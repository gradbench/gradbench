#include "gradbench/evals/ht.hpp"
#include "enzyme.h"
#include "gradbench/main.hpp"
#include <algorithm>

void ht_objective_complicated(const double* const theta, const double* const us,
                              const ht::DataLightMatrix* data, double* err) {
  ht::objective(theta, us, data, err);
}

void d_ht_objective_complicated(double const* theta, double const* d_theta,
                                const double* const        us,
                                const double* const        d_us,
                                const ht::DataLightMatrix* data, double* err,
                                double* d_err) {
  __enzyme_fwddiff(ht_objective_complicated, enzyme_dup, theta, d_theta,
                   enzyme_dup, us, d_us, enzyme_const, data, enzyme_dupnoneed,
                   err, d_err);
}

void compute_ht_complicated_J(const std::vector<double>& theta,
                              const std::vector<double>& us,
                              const ht::DataLightMatrix& data,
                              std::vector<double>*       err,
                              std::vector<double>*       pJ) {
  std::vector<double> d_theta(theta.size());
  std::vector<double> d_err(err->size());
  std::vector<double> d_us(us.size());
  for (int i = 0; i < 2 + theta.size(); i++) {
    if (i >= 2) {
      d_theta[i - 2] = 1;
    } else {
      for (int j = 0; j < d_us.size(); j++) {
        d_us[j] = (1 + i + j) % 2;
      }
    }
    d_ht_objective_complicated(theta.data(), d_theta.data(), us.data(),
                               d_us.data(), &data, err->data(), d_err.data());
    int Jrows = 3 * data.correspondences.size();
    for (int j = 0; j < d_err.size(); j++) {
      (*pJ)[i * Jrows + j] = d_err[j];
    }
    if (i >= 2) {
      d_theta[i - 2] = 0;
    } else {
      for (int j = 0; j < d_us.size(); j++) {
        d_us[j] = 0;
      }
    }
  }
}

void ht_objective_simple(const double* const        theta,
                         const ht::DataLightMatrix* data, double* err) {
  ht::objective(theta, data, err);
}

void d_ht_objective_simple(double const* theta, double const* d_theta,
                           const ht::DataLightMatrix* data, double* err,
                           double* d_err) {
  __enzyme_fwddiff(ht_objective_simple, enzyme_dup, theta, d_theta,
                   enzyme_const, data, enzyme_dupnoneed, err, d_err);
}

void compute_ht_simple_J(const std::vector<double>& theta,
                         const ht::DataLightMatrix& data,
                         std::vector<double>* err, std::vector<double>* pJ) {
  std::vector<double> d_theta(theta.size());
  std::vector<double> d_err(err->size());
  for (int i = 0; i < theta.size(); i++) {
    d_theta[i] = 1;
    d_ht_objective_simple(theta.data(), d_theta.data(), &data, err->data(),
                          d_err.data());
    d_theta[i] = 0;
    int Jrows  = 3 * data.correspondences.size();
    for (int j = 0; j < d_err.size(); j++) {
      (*pJ)[i * Jrows + j] = d_err[j];
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

    if (_input.us.size() == 0) {
      compute_ht_simple_J(_input.theta, _input.data, &_objective,
                          &output.jacobian);
    } else {
      compute_ht_complicated_J(_input.theta, _input.us, _input.data,
                               &_objective, &output.jacobian);
    }
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv,
                      {{"objective", function_main<ht::Objective>},
                       {"jacobian", function_main<Jacobian>}});
}
