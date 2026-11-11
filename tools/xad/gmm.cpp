#include "gradbench/evals/gmm.hpp"
#include "gradbench/main.hpp"
#include <XAD/XAD.hpp>
#include <vector>

class Jacobian : public Function<gmm::Input, gmm::JacOutput> {
  using mode = xad::adj<double>;
  using AD   = mode::active_type;

  mode::tape_type _tape;

  static void read_derivatives(const std::vector<AD>& v,
                               std::vector<double>&   out) {
    out.resize(v.size());
    for (size_t i = 0; i < v.size(); i++) {
      out[i] = xad::derivative(v[i]);
    }
  }

public:
  Jacobian(gmm::Input& input) : Function(input) {}

  void compute(gmm::JacOutput& output) {
    output.d = _input.d;
    output.k = _input.k;
    output.n = _input.n;

    _tape.clearAll();
    std::vector<AD> alpha(_input.alpha.begin(), _input.alpha.end());
    std::vector<AD> mu(_input.mu.begin(), _input.mu.end());
    std::vector<AD> q(_input.q.begin(), _input.q.end());
    std::vector<AD> l(_input.l.begin(), _input.l.end());
    _tape.registerInputs(alpha);
    _tape.registerInputs(mu);
    _tape.registerInputs(q);
    _tape.registerInputs(l);
    _tape.newRecording();

    AD err;
    gmm::objective(_input.d, _input.k, _input.n, alpha.data(), mu.data(),
                   q.data(), l.data(), _input.x.data(), _input.wishart, &err);

    _tape.registerOutput(err);
    xad::derivative(err) = 1.0;
    _tape.computeAdjoints();

    read_derivatives(alpha, output.alpha);
    read_derivatives(mu, output.mu);
    read_derivatives(q, output.q);
    read_derivatives(l, output.l);
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv,
                      {{"objective", function_main<gmm::Objective>},
                       {"jacobian", function_main<Jacobian>}});
}
