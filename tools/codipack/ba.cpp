#include "gradbench/evals/ba.hpp"
#include "gradbench/main.hpp"
#include <codi.hpp>

using Real    = codi::RealReverse;
using RealFwd = codi::RealForward;
using Tape    = typename Real::Tape;

class Jacobian : public Function<ba::Input, ba::JacOutput> {
public:
  Jacobian(ba::Input& input) : Function(input) {}

  void compute(ba::JacOutput& output) {
    output = ba::SparseMat(_input.n, _input.m, _input.p);

    const int           cols = BA_NCAMPARAMS + 3 + 1;
    std::vector<double> gradient(2 * cols);

    std::vector<Real> ad_cams(_input.cams.size());
    std::vector<Real> ad_X(_input.X.size());
    std::vector<Real> ad_w(_input.w.size());
    std::vector<Real> ad_feats(_input.feats.size());

    std::copy(_input.cams.begin(), _input.cams.end(), ad_cams.begin());
    std::copy(_input.X.begin(), _input.X.end(), ad_X.begin());
    std::copy(_input.w.begin(), _input.w.end(), ad_w.begin());
    std::copy(_input.feats.begin(), _input.feats.end(), ad_feats.begin());

    Tape& tape = Real::getTape();

    for (int i = 0; i < _input.p; i++) {
      const int camIdx = _input.obs[i * 2 + 0];
      const int ptIdx  = _input.obs[i * 2 + 1];

      tape.reset();
      tape.setActive();

      Real ad_reproj_err[2];

      for (size_t j = 0; j < BA_NCAMPARAMS; j++) {
        tape.registerInput(ad_cams[camIdx * BA_NCAMPARAMS + j]);
      }
      for (size_t j = 0; j < 3; j++) {
        tape.registerInput(ad_X[ptIdx * 3 + j]);
      }
      tape.registerInput(ad_w[i]);

      ba::computeReprojError<Real>(&ad_cams[camIdx * BA_NCAMPARAMS],
                                   &ad_X[ptIdx * 3], &ad_w[i],
                                   &_input.feats[i * 2], ad_reproj_err);

      tape.registerOutput(ad_reproj_err[0]);
      tape.registerOutput(ad_reproj_err[1]);
      tape.setPassive();

      // Compute first row.
      {
        ad_reproj_err[0].setGradient(1);
        ad_reproj_err[1].setGradient(0);
        tape.evaluate();

        int o = 0;
        for (size_t j = 0; j < BA_NCAMPARAMS; j++) {
          gradient[2 * o++] = ad_cams[camIdx * BA_NCAMPARAMS + j].getGradient();
        }
        for (size_t j = 0; j < 3; j++) {
          gradient[2 * o++] = ad_X[ptIdx * 3 + j].getGradient();
        }
        gradient[2 * o++] = ad_w[i].getGradient();
      }

      // Compute second row.
      {
        tape.clearAdjoints();
        ad_reproj_err[0].setGradient(0);
        ad_reproj_err[1].setGradient(1);
        tape.evaluate();

        int o = 0;
        for (size_t j = 0; j < BA_NCAMPARAMS; j++) {
          gradient[1 + 2 * o++] =
              ad_cams[camIdx * BA_NCAMPARAMS + j].getGradient();
        }
        for (size_t j = 0; j < 3; j++) {
          gradient[1 + 2 * o++] = ad_X[ptIdx * 3 + j].getGradient();
        }
        gradient[1 + 2 * o++] = ad_w[i].getGradient();
      }

      output.insert_reproj_err_block(i, camIdx, ptIdx, gradient.data());
    }

    for (int i = 0; i < _input.p; i++) {
      RealFwd err;
      RealFwd fwd_w = _input.w[i];
      fwd_w.setGradient(1);
      ba::computeZachWeightError<RealFwd>(&fwd_w, &err);
      output.insert_w_err_block(i, err.gradient());
    }
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv,
                      {{"objective", function_main<ba::Objective>},
                       {"jacobian", function_main<Jacobian>}});
}
