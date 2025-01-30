#include "EnzymeHT.h"

#include "adbench/shared/ht_light_matrix.h"

EnzymeHand::EnzymeHand(HandInput& input) : ITest(input) {
  _complicated = _input.us.size() != 0;
  int err_size = 3 * _input.data.correspondences.size();
  int ncols = (_complicated ? 2 : 0) + _input.theta.size();
  _output = { std::vector<double>(err_size), ncols, err_size, std::vector<double>(err_size * ncols) };
}

void EnzymeHand::calculate_objective() {
  if (_complicated) {
    hand_objective(_input.theta.data(), _input.us.data(), &_input.data, _output.objective.data());
  } else {
    hand_objective(_input.theta.data(), &_input.data, _output.objective.data());
  }
}

extern int enzyme_dup;
extern int enzyme_dupnoneed;
extern int enzyme_out;
extern int enzyme_const;
void __enzyme_fwddiff(... ) noexcept;

void hand_objective_complicated(const double* const theta,
                                const double* const us,
                                const HandDataLightMatrix* data,
                                double *err) {
  hand_objective(theta, us, data, err);
}

void d_hand_objective_complicated(double const* theta, double const* d_theta,
                                  const double* const us, const double* const d_us,
                                  const HandDataLightMatrix* data,
                                  double* err, double* d_err) {
  __enzyme_fwddiff(hand_objective_complicated,
                   enzyme_dup, theta, d_theta,
                   enzyme_dup, us, d_us,
                   enzyme_const, data,
                   enzyme_dupnoneed, err, d_err);
}

void compute_hand_complicated_J(const vector<double>& theta,
                                const vector<double>& us,
                                const HandDataLightMatrix& data,
                                vector<double> *err, vector<double> *pJ) {
  std::vector<double> d_theta(theta.size());
  std::vector<double> d_err(err->size());
  std::vector<double> d_us(us.size());
  for (int i = 0; i < 2+theta.size(); i++) {
    if (i >= 2) {
      d_theta[i-2] = 1;
    } else {
      for (int j = 0; j < d_us.size(); j++) {
        d_us[j] = (1+i+j) % 2;
      }
    }
    d_hand_objective_complicated(theta.data(), d_theta.data(),
                                 us.data(), d_us.data(),
                                 &data,
                                 err->data(), d_err.data());
    int Jrows = 3* data.correspondences.size();
    for (int j = 0; j < d_err.size(); j++) {
      (*pJ)[i*Jrows+j] = d_err[j];
    }
    if (i >= 2) {
      d_theta[i-2] = 0;
    } else {
      for (int j = 0; j < d_us.size(); j++) {
        d_us[j] = 0;
      }
    }
  }
}

void hand_objective_simple(const double* const theta,
                           const HandDataLightMatrix* data,
                           double *err) {
  hand_objective(theta, data, err);
}

void d_hand_objective_simple(double const* theta, double const* d_theta,
                             const HandDataLightMatrix* data,
                             double* err, double* d_err) {
  __enzyme_fwddiff(hand_objective_simple,
                   enzyme_dup, theta, d_theta,
                   enzyme_const, data,
                   enzyme_dupnoneed, err, d_err);
}

void compute_hand_simple_J(const vector<double>& theta, const HandDataLightMatrix& data,
                           vector<double> *err, vector<double> *pJ) {
  std::vector<double> d_theta(theta.size());
  std::vector<double> d_err(err->size());
  for (int i = 0; i < theta.size(); i++) {
    d_theta[i] = 1;
    d_hand_objective_simple(theta.data(), d_theta.data(),
                            &data,
                            err->data(), d_err.data());
    d_theta[i] = 0;
    int Jrows = 3* data.correspondences.size();
    for (int j = 0; j < d_err.size(); j++) {
      (*pJ)[i*Jrows+j] = d_err[j];
    }
  }
}

void EnzymeHand::calculate_jacobian() {
  if (_input.us.size() == 0) {
    compute_hand_simple_J(_input.theta,
                          _input.data,
                          &_output.objective,
                          &_output.jacobian);
  } else {
    compute_hand_complicated_J(_input.theta,
                               _input.us,
                               _input.data,
                               &_output.objective,
                               &_output.jacobian);
  }
}
