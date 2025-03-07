#include "gradbench/main.hpp"
#include "gradbench/evals/gmm.hpp"
#include <codi.hpp>

 using Real = codi::RealReverse;
using Tape = typename Real::Tape;

class Jacobian : public Function<gmm::Input, gmm::JacOutput> {
  std::vector<Real> alphas_d;
  std::vector<Real> means_d;
  std::vector<Real> icf_d;

public:
  Jacobian(gmm::Input& input) :
    Function(input),
    alphas_d(_input.k),
    means_d(_input.d*_input.k),
    icf_d((_input.d*(_input.d + 1) / 2)*_input.k) {
    Tape& tape = Real::getTape();
    tape.setActive();

    for (size_t i = 0; i < alphas_d.size(); i++) {
      alphas_d[i] = _input.alphas[i];
      tape.registerInput(alphas_d[i]);
    }
    for (size_t i = 0; i < means_d.size(); i++) {
      means_d[i] = _input.means[i];
      tape.registerInput(means_d[i]);
    }
    for (size_t i = 0; i < icf_d.size(); i++) {
      icf_d[i] = _input.icf[i];
      tape.registerInput(icf_d[i]);
    }
  }

  void compute(gmm::JacOutput& output) {
    int Jcols = (_input.k * (_input.d + 1) * (_input.d + 2)) / 2;
    output.resize(Jcols);

    Tape& tape = Real::getTape();

    Real error;
    gmm::objective(_input.d, _input.k, _input.n,
                   alphas_d.data(),
                   means_d.data(),
                   icf_d.data(),
                   _input.x.data(),
                   _input.wishart,
                   &error);

    tape.registerOutput(error);
    tape.setPassive();
    error.setGradient(1.0);
    tape.evaluate();

    int o = 0;
    for (size_t i = 0; i < alphas_d.size(); i++) {
      output[o++] = alphas_d[i].getGradient();
    }
    for (size_t i = 0; i < means_d.size(); i++) {
      output[o++] = means_d[i].getGradient();
    }
    for (size_t i = 0; i < icf_d.size(); i++) {
      output[o++] = icf_d[i].getGradient();
    }
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv, {
      {"objective", function_main<gmm::Objective>},
      {"jacobian", function_main<Jacobian>}
    });;
}
