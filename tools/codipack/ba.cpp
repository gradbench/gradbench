#include "gradbench/evals/ba.hpp"
#include "codi_impl.hpp"
#include "gradbench/main.hpp"

class Jacobian : public Function<ba::Input, ba::JacOutput>,
                 CoDiReverseRunnerVec<2> {
  using Real    = typename CoDiReverseRunnerVec<2>::Real;
  using RealFwd = typename CoDiForwardRunner::Real;

  const int cols = BA_NCAMPARAMS + 3 + 1;

  std::vector<double> gradient;

  std::vector<Real> ad_cams;
  std::vector<Real> ad_X;
  std::vector<Real> ad_w;
  std::vector<Real> ad_feats;

  Real ad_reproj_err[2];

public:
  Jacobian(ba::Input& input)
      : Function(input), gradient(2 * cols), ad_cams(input.cams.size()),
        ad_X(input.X.size()), ad_w(input.w.size()),
        ad_feats(input.feats.size()) {
    std::copy(_input.cams.begin(), _input.cams.end(), ad_cams.begin());
    std::copy(_input.X.begin(), _input.X.end(), ad_X.begin());
    std::copy(_input.w.begin(), _input.w.end(), ad_w.begin());
    std::copy(_input.feats.begin(), _input.feats.end(), ad_feats.begin());
  }

  void compute(ba::JacOutput& output) {
    output = ba::SparseMat(_input.n, _input.m, _input.p);

    for (int i = 0; i < _input.p; i++) {
      const int camIdx = _input.obs[i * 2 + 0];
      const int ptIdx  = _input.obs[i * 2 + 1];

      codiStartRecording();

      for (size_t j = 0; j < BA_NCAMPARAMS; j++) {
        codiAddInput(ad_cams[camIdx * BA_NCAMPARAMS + j]);
      }
      for (size_t j = 0; j < 3; j++) {
        codiAddInput(ad_X[ptIdx * 3 + j]);
      }
      codiAddInput(ad_w[i]);

      ba::computeReprojError<Real>(&ad_cams[camIdx * BA_NCAMPARAMS],
                                   &ad_X[ptIdx * 3], &ad_w[i],
                                   &_input.feats[i * 2], ad_reproj_err);

      codiAddOutput(ad_reproj_err[0]);
      codiAddOutput(ad_reproj_err[1]);
      codiStopRecording();

      // Compute rows.
      for (int d = 0; d < 2; d += 1) {
        codiSetGradient(ad_reproj_err[d], d, 1.0);
      }

      codiEval();

      for (int d = 0; d < 2; d += 1) {
        int o = 0;
        for (size_t j = 0; j < BA_NCAMPARAMS; j++) {
          gradient[d + 2 * o++] =
              codiGetGradient(ad_cams[camIdx * BA_NCAMPARAMS + j], d);
        }
        for (size_t j = 0; j < 3; j++) {
          gradient[d + 2 * o++] = codiGetGradient(ad_X[ptIdx * 3 + j], d);
        }
        gradient[d + 2 * o++] = codiGetGradient(ad_w[i], d);
      }

      output.insert_reproj_err_block(i, camIdx, ptIdx, gradient.data());

      codiCleanup();
    }

    CoDiForwardRunner fr{};
    for (int i = 0; i < _input.p; i++) {
      RealFwd err;
      RealFwd fwd_w = _input.w[i];

      fr.codiSetGradient(fwd_w, 1.0);
      ba::computeZachWeightError<RealFwd>(&fwd_w, &err);
      output.insert_w_err_block(i, fr.codiGetGradient(err));
    }
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv,
                      {{"objective", function_main<ba::Objective>},
                       {"jacobian", function_main<Jacobian>}});
}
