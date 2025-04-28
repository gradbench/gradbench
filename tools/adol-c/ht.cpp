// This uses ADOL-Cs "traceless forward differentiation" (forward mode
// without a tape), in vector mode. It obtains 2.5x speedup over
// reverse mode.

#include "gradbench/evals/ht.hpp"
#include "gradbench/main.hpp"

#include <adolc/adtl.h>

typedef adtl::adouble adouble;

void compute_ht_complicated_J(const std::vector<double>& theta,
                              const std::vector<double>& us,
                              const ht::DataLightMatrix& data,
                              std::vector<double>&       J) {
  size_t               Jrows = 3 * data.correspondences.size();
  std::vector<adouble> a_theta(theta.size());
  std::vector<adouble> a_us(us.size());
  std::vector<adouble> a_err(Jrows);

  std::copy(theta.begin(), theta.end(), a_theta.begin());
  std::copy(us.begin(), us.end(), a_us.begin());

  for (size_t i = 0; i < 2 + theta.size(); i++) {
    if (i >= 2) {
      a_theta[i - 2].setADValue(i, 1);
    } else {
      for (size_t j = 0; j < a_us.size(); j++) {
        double d = (1 + i + j) % 2;
        a_us[j].setADValue(i, d);
      }
    }
  }

  ht::objective<adouble>(a_theta.data(), a_us.data(), &data, a_err.data());

  for (size_t i = 0; i < 2 + theta.size(); i++) {
    for (size_t j = 0; j < Jrows; j++) {
      J[i * Jrows + j] = a_err[j].getADValue()[i];
    }
  }
}

void compute_ht_simple_J(std::vector<double>&       theta,
                         const ht::DataLightMatrix& data,
                         std::vector<double>&       J) {
  size_t               Jrows = 3 * data.correspondences.size();
  std::vector<adouble> a_theta(theta.size());
  std::vector<adouble> a_err(Jrows);

  std::copy(theta.begin(), theta.end(), a_theta.begin());

  for (size_t i = 0; i < theta.size(); i++) {
    a_theta[i].setADValue(i, 1);
  }

  ht::objective<adouble>(a_theta.data(), &data, a_err.data());

  for (size_t i = 0; i < theta.size(); i++) {
    for (size_t j = 0; j < a_err.size(); j++) {
      J[i * Jrows + j] = a_err[j].getADValue()[i];
    }
  }
}

class Jacobian : public Function<ht::Input, ht::JacOutput> {
  bool _complicated = false;

public:
  Jacobian(ht::Input& input)
      : Function(input), _complicated(input.us.size() != 0) {
    adtl::setNumDir(2 + _input.theta.size());
  }

  void compute(ht::JacOutput& output) {
    int err_size          = 3 * _input.data.correspondences.size();
    int ncols             = (_complicated ? 2 : 0) + _input.theta.size();
    output.jacobian_ncols = ncols;
    output.jacobian_nrows = err_size;
    output.jacobian.resize(err_size * ncols);

    if (_complicated) {
      compute_ht_complicated_J(_input.theta, _input.us, _input.data,
                               output.jacobian);
    } else {
      compute_ht_simple_J(_input.theta, _input.data, output.jacobian);
    }
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv,
                      {{"objective", function_main<ht::Objective>},
                       {"jacobian", function_main<Jacobian>}});
}
