#include "gradbench/evals/gmm.hpp"
#include "codi_impl.hpp"
#include "gradbench/main.hpp"

class Jacobian : public Function<gmm::Input, gmm::JacOutput>,
                 CoDiReverseRunner {
  using Real = typename CoDiReverseRunner::Real;

  std::vector<Real> alphas_d;
  std::vector<Real> mu_d;
  std::vector<Real> q_d;
  std::vector<Real> l_d;

  Real error;

public:
  Jacobian(gmm::Input& input)
    : Function(input), alphas_d(_input.k), mu_d(_input.d * _input.k),
      q_d(_input.k * _input.d * _input.d),
      l_d(_input.k * _input.d * (_input.d-1) / 2),
      error() {

    for (size_t i = 0; i < alphas_d.size(); i++) {
      alphas_d[i] = _input.alphas[i];
    }
    for (size_t i = 0; i < mu_d.size(); i++) {
      mu_d[i] = _input.mu[i];
    }
    for (size_t i = 0; i < q_d.size(); i++) {
      q_d[i] = _input.q[i];
    }
    for (size_t i = 0; i < l_d.size(); i++) {
      l_d[i] = _input.l[i];
    }
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

    codiStartRecording();

    for (size_t i = 0; i < alphas_d.size(); i++) {
      codiAddInput(alphas_d[i]);
    }
    for (size_t i = 0; i < mu_d.size(); i++) {
      codiAddInput(mu_d[i]);
    }
    for (size_t i = 0; i < q_d.size(); i++) {
      codiAddInput(q_d[i]);
    }
    for (size_t i = 0; i < l_d.size(); i++) {
      codiAddInput(l_d[i]);
    }

    gmm::objective(_input.d, _input.k, _input.n, alphas_d.data(),
                   mu_d.data(), q_d.data(), l_d.data(), _input.x.data(),
                   _input.wishart, &error);

    codiAddOutput(error);
    codiStopRecording();

    codiSetGradient(error, 1.0);
    codiEval();

    for (size_t i = 0; i < alphas_d.size(); i++) {
      output.alpha[i] = codiGetGradient(alphas_d[i]);
    }
    for (size_t i = 0; i < mu_d.size(); i++) {
      output.mu[i] = codiGetGradient(mu_d[i]);
    }
    for (size_t i = 0; i < q_d.size(); i++) {
      output.q[i] = codiGetGradient(q_d[i]);
    }
    for (size_t i = 0; i < l_d.size(); i++) {
      output.l[i] = codiGetGradient(l_d[i]);
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
