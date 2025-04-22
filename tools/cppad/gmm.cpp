#include <algorithm>
#include <vector>

#include "gradbench/evals/gmm.hpp"
#include "gradbench/main.hpp"

#include <cppad/cppad.hpp>

typedef CppAD::AD<double> ADdouble;

class Jacobian : public Function<gmm::Input, gmm::JacOutput> {
private:
  std::vector<double>   _input_flat;
  std::vector<ADdouble> _X;

public:
  Jacobian(gmm::Input& input) : Function(input) {
    _input_flat.insert(_input_flat.end(), _input.alphas.begin(),
                       _input.alphas.end());
    _input_flat.insert(_input_flat.end(), _input.means.begin(),
                       _input.means.end());
    _input_flat.insert(_input_flat.end(), _input.icf.begin(), _input.icf.end());

    _X.resize(_input_flat.size());

    std::copy(_input_flat.begin(), _input_flat.end(), _X.data());
  }

  void compute(gmm::JacOutput& output) {
    int Jcols = (_input.k * (_input.d + 1) * (_input.d + 2)) / 2;
    output.resize(Jcols);

    ADdouble* aalphas = &_X[0];
    ADdouble* ameans  = aalphas + _input.alphas.size();
    ADdouble* aicf    = ameans + _input.means.size();

    CppAD::Independent(_X);

    std::vector<ADdouble> Y(1);

    gmm::objective<ADdouble>(_input.d, _input.k, _input.n, aalphas, ameans,
                             aicf, _input.x.data(), _input.wishart, &Y[0]);

    CppAD::ADFun<double> f(_X, Y);

    output = f.Jacobian(_input_flat);
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv,
                      {{"objective", function_main<gmm::Objective>},
                       {"jacobian", function_main<Jacobian>}});
  ;
}
