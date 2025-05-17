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
    _input_flat.insert(_input_flat.end(), _input.alpha.begin(),
                       _input.alpha.end());
    _input_flat.insert(_input_flat.end(), _input.mu.begin(), _input.mu.end());
    _input_flat.insert(_input_flat.end(), _input.q.begin(), _input.q.end());
    _input_flat.insert(_input_flat.end(), _input.l.begin(), _input.l.end());

    _X.resize(_input_flat.size());

    std::copy(_input_flat.begin(), _input_flat.end(), _X.data());
  }

  void compute(gmm::JacOutput& output) {
    const int l_sz = _input.d * (_input.d - 1) / 2;

    output.d = _input.d;
    output.k = _input.k;
    output.n = _input.n;

    output.alpha.resize(output.k);
    output.mu.resize(output.k * output.d);
    output.q.resize(output.k * output.d);
    output.l.resize(output.k * l_sz);

    ADdouble* aalpha = &_X[0];
    ADdouble* amu    = aalpha + _input.alpha.size();
    ADdouble* aq     = amu + _input.mu.size();
    ADdouble* al     = aq + _input.q.size();

    CppAD::Independent(_X);

    std::vector<ADdouble> Y(1);

    gmm::objective<ADdouble>(_input.d, _input.k, _input.n, aalpha, amu, aq, al,
                             _input.x.data(), _input.wishart, &Y[0]);

    CppAD::ADFun<double> f(_X, Y);

    std::vector<double> J   = f.Jacobian(_input_flat);
    int                 off = 0;
    std::copy(J.begin() + off, J.begin() + off + _input.alpha.size(),
              output.alpha.begin());
    off += _input.alpha.size();
    std::copy(J.begin() + off, J.begin() + off + _input.mu.size(),
              output.mu.begin());
    off += _input.mu.size();
    std::copy(J.begin() + off, J.begin() + off + _input.q.size(),
              output.q.begin());
    off += _input.q.size();
    std::copy(J.begin() + off, J.begin() + off + _input.l.size(),
              output.l.begin());
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv,
                      {{"objective", function_main<gmm::Objective>},
                       {"jacobian", function_main<Jacobian>}});
  ;
}
