#include "gradbench/evals/ba.hpp"
#include "gradbench/main.hpp"
#include <XAD/XAD.hpp>
#include <vector>

class Jacobian : public Function<ba::Input, ba::JacOutput> {
  // Vector adjoint mode with 2 directions: both rows of the 2x15
  // reprojection error block come out of one reverse sweep.
  using mode  = xad::adj<double, 2>;
  using AD    = mode::active_type;
  using FwdAD = xad::fwd<double>::active_type;

  static constexpr int cols = BA_NCAMPARAMS + 3 + 1;

  mode::tape_type     _tape;
  std::vector<double> _gradient;

public:
  Jacobian(ba::Input& input) : Function(input), _gradient(2 * cols) {}

  void compute(ba::JacOutput& output) {
    output = ba::SparseMat(_input.n, _input.m, _input.p);

    for (int i = 0; i < _input.p; i++) {
      const int camIdx = _input.obs[i * 2 + 0];
      const int ptIdx  = _input.obs[i * 2 + 1];

      // Inputs of this observation: camera, point, weight.
      _tape.clearAll();
      AD in[cols];
      for (int j = 0; j < BA_NCAMPARAMS; j++) {
        in[j] = _input.cams[camIdx * BA_NCAMPARAMS + j];
      }
      for (int j = 0; j < 3; j++) {
        in[BA_NCAMPARAMS + j] = _input.X[ptIdx * 3 + j];
      }
      in[cols - 1] = _input.w[i];
      _tape.registerInputs(in, in + cols);
      _tape.newRecording();

      AD err[2];
      ba::computeReprojError<AD>(&in[0], &in[BA_NCAMPARAMS], &in[cols - 1],
                                 &_input.feats[i * 2], err);

      _tape.registerOutput(err[0]);
      _tape.registerOutput(err[1]);
      xad::derivative(err[0])[0] = 1.0;
      xad::derivative(err[1])[1] = 1.0;
      _tape.computeAdjoints();

      for (int d = 0; d < 2; d++) {
        for (int o = 0; o < cols; o++) {
          _gradient[d + 2 * o] = xad::derivative(in[o])[d];
        }
      }

      output.insert_reproj_err_block(i, camIdx, ptIdx, _gradient.data());
    }

    for (int i = 0; i < _input.p; i++) {
      FwdAD w = _input.w[i];
      FwdAD err;
      xad::derivative(w) = 1.0;
      ba::computeZachWeightError<FwdAD>(&w, &err);
      output.insert_w_err_block(i, xad::derivative(err));
    }
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv,
                      {{"objective", function_main<ba::Objective>},
                       {"jacobian", function_main<Jacobian>}});
}
