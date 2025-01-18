#include "EnzymeBA.h"
#include "adbench/shared/ba.h"
#include <algorithm>

void EnzymeBA::prepare(BAInput&& input) {
  _input = input;
  _output = {
    std::vector<double>(2 * _input.p),
    std::vector<double>(_input.p),
    BASparseMat(_input.n, _input.m, _input.p)
  };
  int n_new_cols = BA_NCAMPARAMS + 3 + 1;
  _reproj_err_d = std::vector<double>(2 * n_new_cols);
}

BAOutput EnzymeBA::output() {
  return _output;
}

void EnzymeBA::calculate_objective(int times) {
  for (int i = 0; i < times; ++i) {
    ba_objective(_input.n, _input.m, _input.p,
                 _input.cams.data(), _input.X.data(), _input.w.data(),
                 _input.obs.data(), _input.feats.data(),
                 _output.reproj_err.data(), _output.w_err.data());
  }
}

extern int enzyme_dup;
extern int enzyme_dupnoneed;
extern int enzyme_out;
extern int enzyme_const;
void __enzyme_autodiff(... ) noexcept;

void d_computeReprojError(double const *cam, double *d_cam,
                          double const *X, double *d_X,
                          double const *w, double *d_w,
                          double const *feat,
                          double *err, double *d_err) {
  __enzyme_autodiff(computeReprojError<double>,
                    enzyme_dup, cam, d_cam,
                    enzyme_dup, X, d_X,
                    enzyme_dup, w, d_w,

                    enzyme_const, feat,
                    enzyme_dupnoneed, err, d_err);
}

void d_computeZachWeightError(double const *w, double *d_w,
                              double *w_err, double *d_w_err) {
  __enzyme_autodiff(computeZachWeightError<double>,
                    enzyme_dup, w, d_w,
                    enzyme_dupnoneed, w_err, d_w_err);
}

void compute_ba_J(int n, int m, int p,
                  double *cams, double *X, double *w, int *obs, double *feats,
                  double *reproj_err, double *w_err, BASparseMat *J) {
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

void EnzymeBA::calculate_jacobian(int times) {
  for (int i = 0; i < times; ++i) {
    _output.J.clear();
    compute_ba_J(_input.n, _input.m, _input.p,
                 _input.cams.data(), _input.X.data(), _input.w.data(),
                 _input.obs.data(), _input.feats.data(),
                 _output.reproj_err.data(), _output.w_err.data(),
                 &_output.J);
  }
}
