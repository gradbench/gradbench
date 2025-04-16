#include "gradbench/main.hpp"
#include "gradbench/evals/ba.hpp"
#include "ad.hpp"

using adjoint_t = ad::adjoint_t<double>;
using adjoint   = ad::adjoint<double>;

class Jacobian : public Function<ba::Input, ba::JacOutput> {
public:
  Jacobian(ba::Input& input) :
    Function(input) {
    adjoint::global_tape = adjoint::tape_t::create();
  }

  void compute(ba::JacOutput& output) {
    output = ba::SparseMat(_input.n, _input.m, _input.p);

    const int cols = BA_NCAMPARAMS + 3 + 1;
    std::vector<double> gradient(2 * cols);

    std::vector<adjoint_t> ad_cams(_input.cams.size());
    std::vector<adjoint_t> ad_X(_input.X.size());
    std::vector<adjoint_t> ad_w(_input.w.size());
    std::vector<adjoint_t> ad_feats(_input.feats.size());

    std::copy(_input.cams.begin(), _input.cams.end(), ad_cams.begin());
    std::copy(_input.X.begin(), _input.X.end(), ad_X.begin());
    std::copy(_input.w.begin(), _input.w.end(), ad_w.begin());
    std::copy(_input.feats.begin(), _input.feats.end(), ad_feats.begin());

    for (int i = 0; i < _input.p; i++) {
      const int camIdx = _input.obs[i * 2 + 0];
      const int ptIdx = _input.obs[i * 2 + 1];

      adjoint::global_tape->reset();

      adjoint_t ad_reproj_err[2];

      for (size_t j = 0; j < BA_NCAMPARAMS; j++) {
        adjoint::global_tape->register_variable(ad_cams[camIdx*BA_NCAMPARAMS+j]);
      }
      for (size_t j = 0; j < 3; j++) {
        adjoint::global_tape->register_variable(ad_X[ptIdx*3 + j]);
      }
      adjoint::global_tape->register_variable(ad_w[i]);

      ba::computeReprojError<adjoint_t>(&ad_cams[camIdx*BA_NCAMPARAMS],
                                        &ad_X[ptIdx * 3],
                                        &ad_w[i],
                                        &_input.feats[i * 2],
                                        ad_reproj_err);

      // Compute first row.
      {
        ad::derivative(ad_reproj_err[0]) = 1;
        ad::derivative(ad_reproj_err[1]) = 0;
        adjoint::global_tape->interpret_adjoint();

        int o = 0;
        for (size_t j = 0; j < BA_NCAMPARAMS; j++) {
          gradient[2 * o++] = ad::derivative(ad_cams[camIdx*BA_NCAMPARAMS+j]);
        }
        for (size_t j = 0; j < 3; j++) {
          gradient[2 * o++] = ad::derivative(ad_X[ptIdx*3 + j]);
        }
        gradient[2 * o++] = ad::derivative(ad_w[i]);
      }

      // Compute second row.
      {
        adjoint::global_tape->zero_adjoints();
        ad::derivative(ad_reproj_err[0]) = 0;
        ad::derivative(ad_reproj_err[1]) = 1;
        adjoint::global_tape->interpret_adjoint();

        int o = 0;
        for (size_t j = 0; j < BA_NCAMPARAMS; j++) {
          gradient[1+ 2*o++] = ad::derivative(ad_cams[camIdx*BA_NCAMPARAMS+j]);
        }
        for (size_t j = 0; j < 3; j++) {
          gradient[1+ 2*o++] = ad::derivative(ad_X[ptIdx*3 + j]);
        }
        gradient[1+ 2*o++] = ad::derivative(ad_w[i]);
      }

      output.insert_reproj_err_block(i, camIdx, ptIdx, gradient.data());
    }

    for (int i = 0; i < _input.p; i++) {
      adjoint::global_tape->reset();
      adjoint_t err;
      adjoint_t w = _input.w[i];
      adjoint::global_tape->register_variable(w);
      ba::computeZachWeightError<adjoint_t>(&w, &err);
      ad::derivative(err) = 1;
      adjoint::global_tape->interpret_adjoint();
      output.insert_w_err_block(i, ad::derivative(w));
    }
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv, {
      {"objective", function_main<ba::Objective>},
      {"jacobian", function_main<Jacobian>}
    });
}
