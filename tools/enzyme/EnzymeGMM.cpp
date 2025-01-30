#include "EnzymeGMM.h"
#include "adbench/shared/gmm.h"
#include <algorithm>

EnzymeGMM::EnzymeGMM(GMMInput& input) : ITest(input) {
  int Jcols = (_input.k * (_input.d + 1) * (_input.d + 2)) / 2;
  _output = { 0,  std::vector<double>(Jcols) };
}

void EnzymeGMM::calculate_objective() {
  gmm_objective(_input.d,
                _input.k,
                _input.n,

                _input.alphas.data(),
                _input.means.data(),
                _input.icf.data(),

                _input.x.data(),
                _input.wishart,
                &_output.objective);
}

extern int enzyme_dup;
extern int enzyme_dupnoneed;
extern int enzyme_out;
extern int enzyme_const;
void __enzyme_autodiff(... ) noexcept;

void EnzymeGMM::calculate_jacobian() {
  _output.gradient.resize(_input.alphas.size()+
                          _input.means.size()+
                          _input.icf.size());


  double* d_alphas = _output.gradient.data();
  double* d_means = d_alphas + _input.alphas.size();
  double* d_icf = d_means + _input.means.size();

  std::fill(_output.gradient.begin(), _output.gradient.end(), 0);
  double d_err = 1;
  __enzyme_autodiff
    (gmm_objective<double>,
     enzyme_const, _input.d,
     enzyme_const, _input.k,
     enzyme_const, _input.n,

     enzyme_dup, _input.alphas.data(), d_alphas,

     enzyme_dup, _input.means.data(), d_means,

     enzyme_dup, _input.icf.data(), d_icf,

     enzyme_const, _input.x.data(),
     enzyme_const, _input.wishart,
     enzyme_dupnoneed, &_output.objective, &d_err);
}
