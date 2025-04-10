#include <algorithm>
#include "gradbench/main.hpp"
#include "gradbench/evals/ba.hpp"
#include "enzyme.h"

void d_computeReprojError(double const *cam, double *d_cam,
                          double const *X, double *d_X,
                          double const *w, double *d_w,
                          double const *feat,
                          double *err, double *d_err) {
  __enzyme_autodiff(ba::computeReprojError<double>,
                    enzyme_dup, cam, d_cam,
                    enzyme_dup, X, d_X,
                    enzyme_dup, w, d_w,

                    enzyme_const, feat,
                    enzyme_dupnoneed, err, d_err);
}

void d_computeZachWeightError(double const *w, double *d_w,
                              double *w_err, double *d_w_err) {
  __enzyme_autodiff(ba::computeZachWeightError<double>,
                    enzyme_dup, w, d_w,
                    enzyme_dupnoneed, w_err, d_w_err);
}

void compute_ba_J(int n, int m, int p,
                  double *cams, double *X, double *w, int *obs, double *feats,
                  double *reproj_err, double *w_err, ba::SparseMat *J) {
  const int cols = BA_NCAMPARAMS + 3 + 1;
  std::vector<double> gradient(2 * cols);
  std::vector<double> row(cols);
  double* d_cam = row.data();
  double* d_X = d_cam + BA_NCAMPARAMS;
  double* d_w = d_X + 3;

  for (int i = 0; i < p; i++) {
    const int camIdx = obs[i * 2 + 0];
    const int ptIdx = obs[i * 2 + 1];

    // Compute first row.
    {
      std::fill(row.begin(), row.end(), 0);

      double d_reproj_err[2] = {1, 0};
      d_computeReprojError(&cams[camIdx * BA_NCAMPARAMS], d_cam,
                           &X[ptIdx * 3], d_X,
                           &w[i], d_w,
                           &feats[i * 2],
                           &reproj_err[i * 2], d_reproj_err);
      for (int j = 0; j < cols; j++) {
        gradient[2 * j] = row[j];
      }
    }

    // Compute second row.
    {
      std::fill(row.begin(), row.end(), 0);

      double d_reproj_err[2] = {0, 1};
      d_computeReprojError(&cams[camIdx * BA_NCAMPARAMS], d_cam,
                           &X[ptIdx * 3], d_X,
                           &w[i], d_w,
                           &feats[i * 2],
                           &reproj_err[i * 2], d_reproj_err);
      for (int j = 0; j < cols; j++) {
        gradient[2 * j + 1] = row[j];
      }
    }

    J->insert_reproj_err_block(i, camIdx, ptIdx, gradient.data());
  }

  for (int i = 0; i < p; i++) {
    double d_w = 0;
    double d_w_err = 1;
    double err;
    d_computeZachWeightError(&w[i], &d_w, &err, &d_w_err);
    J->insert_w_err_block(i, d_w);
  }
}

class Jacobian : public Function<ba::Input, ba::JacOutput> {
public:
  Jacobian(ba::Input& input) : Function(input) {}
  void compute(ba::JacOutput& output) {
    output = ba::SparseMat(_input.n, _input.m, _input.p);
    std::vector<double> reproj_err(2 * _input.p);
    std::vector<double> w_err(_input.p);
    compute_ba_J(_input.n, _input.m, _input.p,
                 _input.cams.data(), _input.X.data(), _input.w.data(),
                 _input.obs.data(), _input.feats.data(),
                 reproj_err.data(), w_err.data(),
                 &output);
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv, {
      {"objective", function_main<ba::Objective>},
      {"jacobian", function_main<Jacobian>}
    });
}
