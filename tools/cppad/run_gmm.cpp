#include <algorithm>
#include <vector>

#include "gradbench/main.hpp"
#include "gradbench/evals/gmm.hpp"

#include <cppad/cppad.hpp>

typedef CppAD::AD<double> ADdouble;

class Jacobian : public Function<gmm::Input, gmm::JacOutput> {
private:
  std::vector<double> _input_flat;
  CppAD::ADFun<double> *_tape;

public:
  Jacobian(gmm::Input& input) : Function(input) {
    _input_flat.insert(_input_flat.end(), _input.alphas.begin(), _input.alphas.end());
    _input_flat.insert(_input_flat.end(), _input.means.begin(), _input.means.end());
    _input_flat.insert(_input_flat.end(), _input.icf.begin(), _input.icf.end());

    std::vector<ADdouble> X(_input_flat.size());

    ADdouble* aalphas = &X[0];
    ADdouble* ameans = aalphas + _input.alphas.size();
    ADdouble* aicf = ameans + _input.means.size();

    std::copy(_input_flat.begin(), _input_flat.end(), X.data());

    CppAD::Independent(X);

    std::vector<ADdouble> Y(1);

    gmm::objective<ADdouble>(_input.d, _input.k, _input.n,
                             aalphas, ameans, aicf,
                             _input.x.data(), _input.wishart, &Y[0]);

    _tape = new CppAD::ADFun<double>(X, Y);

    _tape->optimize();
  }

  void compute(gmm::JacOutput& output) {
    int Jcols = (_input.k * (_input.d + 1) * (_input.d + 2)) / 2;
    output.resize(Jcols);

    output = _tape->Jacobian(_input_flat);
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv, {
      {"objective", function_main<gmm::Objective>},
      {"jacobian", function_main<Jacobian>}
    });;
}
