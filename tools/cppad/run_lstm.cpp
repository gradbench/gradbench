#include <algorithm>
#include "gradbench/main.hpp"
#include "gradbench/evals/lstm.hpp"
#include <cppad/cppad.hpp>

typedef CppAD::AD<double> ADdouble;

class Jacobian : public Function<lstm::Input, lstm::JacOutput> {
  std::vector<double> _input_flat;
  CppAD::ADFun<double> *_tape;

public:
  Jacobian(lstm::Input& input) : Function(input) {
    _input_flat.insert(_input_flat.end(), _input.main_params.begin(), _input.main_params.end());
    _input_flat.insert(_input_flat.end(), _input.extra_params.begin(), _input.extra_params.end());

    std::vector<ADdouble> astate(_input.state.size());
    std::vector<ADdouble> asequence(_input.sequence.size());
    astate.insert(astate.begin(), _input.state.begin(), _input.state.end());
    asequence.insert(asequence.begin(), _input.sequence.begin(), _input.sequence.end());

    std::vector<ADdouble> X(_input_flat.size());
    std::copy(_input_flat.begin(), _input_flat.end(), X.data());
    ADdouble* amain_params = &X[0];
    ADdouble* aextra_params = amain_params + _input.main_params.size();

    CppAD::Independent(X);

    std::vector<ADdouble> Y(1);

    lstm::objective(input.l, input.c, input.b,
                    amain_params, aextra_params,
                    astate.data(), asequence.data(),
                    &Y[0]);

    _tape = new CppAD::ADFun<double>(X, Y);

    _tape->optimize();
  }

  void compute(lstm::JacOutput& output) {
    output.resize(8 * _input.l * _input.b + 3 * _input.b);
    output = _tape->Jacobian(_input_flat);
  }
};


int main(int argc, char* argv[]) {
  return generic_main(argc, argv, {
      {"objective", function_main<lstm::Objective>},
      {"jacobian", function_main<Jacobian>}
    });;
}
