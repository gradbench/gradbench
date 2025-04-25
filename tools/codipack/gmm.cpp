#include "gradbench/evals/gmm.hpp"
#include "codi_impl.hpp"
#include "gradbench/main.hpp"

class Jacobian : public Function<gmm::Input, gmm::JacOutput>,
                 CoDiReverseRunner {
  using Real = typename CoDiReverseRunner::Real;

  std::vector<Real> alphas_d;
  std::vector<Real> means_d;
  std::vector<Real> icf_d;

  Real error;

public:
  Jacobian(gmm::Input& input)
      : Function(input), alphas_d(_input.k), means_d(_input.d * _input.k),
        icf_d((_input.d * (_input.d + 1) / 2) * _input.k), error() {

    for (size_t i = 0; i < alphas_d.size(); i++) {
      alphas_d[i] = _input.alphas[i];
    }
    for (size_t i = 0; i < means_d.size(); i++) {
      means_d[i] = _input.means[i];
    }
    for (size_t i = 0; i < icf_d.size(); i++) {
      icf_d[i] = _input.icf[i];
    }
  }

  void compute(gmm::JacOutput& output) {
    int Jcols = (_input.k * (_input.d + 1) * (_input.d + 2)) / 2;
    output.resize(Jcols);

    codiStartRecording();

    for (size_t i = 0; i < alphas_d.size(); i++) {
      codiAddInput(alphas_d[i]);
    }
    for (size_t i = 0; i < means_d.size(); i++) {
      codiAddInput(means_d[i]);
    }
    for (size_t i = 0; i < icf_d.size(); i++) {
      codiAddInput(icf_d[i]);
    }

    gmm::objective(_input.d, _input.k, _input.n, alphas_d.data(),
                   means_d.data(), icf_d.data(), _input.x.data(),
                   _input.wishart, &error);

    codiAddOutput(error);
    codiStopRecording();

    codiSetGradient(error, 1.0);
    codiEval();

    int o = 0;
    for (size_t i = 0; i < alphas_d.size(); i++) {
      output[o++] = codiGetGradient(alphas_d[i]);
    }
    for (size_t i = 0; i < means_d.size(); i++) {
      output[o++] = codiGetGradient(means_d[i]);
    }
    for (size_t i = 0; i < icf_d.size(); i++) {
      output[o++] = codiGetGradient(icf_d[i]);
    }

    codiCleanup();
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv,
                      {{"objective", function_main<gmm::Objective>},
                       {"jacobian", function_main<Jacobian>}});
  ;
}
