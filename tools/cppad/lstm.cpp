#include "gradbench/evals/lstm.hpp"
#include "gradbench/main.hpp"
#include <algorithm>
#include <cppad/cppad.hpp>

typedef CppAD::AD<double> ADdouble;

class Jacobian : public Function<lstm::Input, lstm::JacOutput> {
  std::vector<double>   _input_flat;
  std::vector<ADdouble> _X, _Y;
  std::vector<ADdouble> _astate, _asequence;

public:
  Jacobian(lstm::Input& input) : Function(input), _Y(1) {
    _input_flat.insert(_input_flat.end(), _input.main_params.begin(),
                       _input.main_params.end());
    _input_flat.insert(_input_flat.end(), _input.extra_params.begin(),
                       _input.extra_params.end());

    _astate.insert(_astate.begin(), _input.state.begin(), _input.state.end());
    _asequence.insert(_asequence.begin(), _input.sequence.begin(),
                      _input.sequence.end());

    _X.resize(_input_flat.size());
    std::copy(_input_flat.begin(), _input_flat.end(), _X.data());
  }

  void compute(lstm::JacOutput& output) {
    output.resize(8 * _input.l * _input.b + 3 * _input.b);

    ADdouble* amain_params  = &_X[0];
    ADdouble* aextra_params = amain_params + _input.main_params.size();

    CppAD::Independent(_X);
    lstm::objective(_input.l, _input.c, _input.b, amain_params, aextra_params,
                    _astate.data(), _asequence.data(), &_Y[0]);
    CppAD::ADFun<double> f(_X, _Y);
    output = f.Jacobian(_input_flat);
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv,
                      {{"objective", function_main<lstm::Objective>},
                       {"jacobian", function_main<Jacobian>}});
  ;
}
