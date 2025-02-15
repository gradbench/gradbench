// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// Mainly derived from:
//  https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/cpp/shared/ba.h
//  https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/cpp/shared/BAData.h

#pragma once

#include <vector>
#include <cmath>
#include "adbench/shared/matrix.h"
#include "gradbench/main.hpp"
#include "json.hpp"

#define BA_NCAMPARAMS 11
#define BA_ROT_IDX 0
#define BA_C_IDX 3
#define BA_F_IDX 6
#define BA_X0_IDX 7
#define BA_RAD_IDX 9

namespace ba {
// cam: 11 camera in format [r1 r2 r3 C1 C2 C3 f u0 v0 k1 k2]
//            r1, r2, r3 are angle - axis rotation parameters(Rodrigues)
//            [C1 C2 C3]' is the camera center
//            f is the focal length in pixels
//            [u0 v0]' is the principal point
//            k1, k2 are radial distortion parameters
// X: 3 point
// feats: 2 feature (x,y coordinates)
// reproj_err: 2
// projection function:
// Xcam = R * (X - C)
// distorted = radial_distort(projective2euclidean(Xcam), radial_parameters)
// proj = distorted * f + principal_point
// err = sqsum(proj - measurement)
template<typename T>
void computeReprojError(
                        const T* const cam,
                        const T* const X,
                        const T* const w,
                        const double* const feat,
                        T *err);

// w: 1
// w_err: 1
template<typename T>
void computeZachWeightError(const T* const w, T* err);

// n number of cameras
// m number of points
// p number of observations
// cams: 11*n cameras in format [r1 r2 r3 C1 C2 C3 f u0 v0 k1 k2]
//            r1, r2, r3 are angle - axis rotation parameters(Rodrigues)
//            [C1 C2 C3]' is the camera center
//            f is the focal length in pixels
//            [u0 v0]' is the principal point
//            k1, k2 are radial distortion parameters
// X: 3*m points
// obs: 2*p observations (pairs cameraIdx, pointIdx)
// feats: 2*p features (x,y coordinates corresponding to observations)
// reproj_err: 2*p errors of observations
// w_err: p weight "error" terms
// projection function:
// Xcam = R * (X - C)
// distorted = radial_distort(projective2euclidean(Xcam), radial_parameters)
// proj = distorted * f + principal_point
// err = sqsum(proj - measurement)
template<typename T>
void objective(int n, int m, int p,
               const T* const cams,
               const T* const X,
               const T* const w,
               const int* const obs,
               const double* const feats,
               T* reproj_err,
               T* w_err);

// rot: 3 rotation parameters
// pt: 3 point to be rotated
// rotatedPt: 3 rotated point
// this is an efficient evaluation (part of
// the Ceres implementation)
// easy to understand calculation in matlab:
//  theta = sqrt(sum(w. ^ 2));
//  n = w / theta;
//  n_x = au_cross_matrix(n);
//  R = eye(3) + n_x*sin(theta) + n_x*n_x*(1 - cos(theta));
template<typename T>
void rodrigues_rotate_point(const T* const rot,
                            const T* const pt,
                            T *rotatedPt);

////////////////////////////////////////////////////////////
//////////////////// Definitions ///////////////////////////
////////////////////////////////////////////////////////////

template<typename T>
T sqsum(int n, const T* const x) {
  T res = 0;
  for (int i = 0; i < n; i++)
    res = res + x[i] * x[i];
  return res;
}

template<typename T>
void rodrigues_rotate_point(const T* const rot,
                            const T* const pt,
                            T *rotatedPt) {
  T sqtheta = sqsum(3, rot);
  if (sqtheta != 0) {
    T theta, costheta, sintheta, theta_inverse,
      w[3], w_cross_pt[3], tmp;

    theta = sqrt(sqtheta);
    costheta = cos(theta);
    sintheta = sin(theta);
    theta_inverse = 1.0 / theta;

    for (int i = 0; i < 3; i++)
      w[i] = rot[i] * theta_inverse;

    cross(w, pt, w_cross_pt);

    tmp = (w[0] * pt[0] + w[1] * pt[1] + w[2] * pt[2]) *
      (1. - costheta);

    for (int i = 0; i < 3; i++)
      rotatedPt[i] = pt[i] * costheta + w_cross_pt[i] * sintheta + w[i] * tmp;
  } else {
    T rot_cross_pt[3];
    cross(rot, pt, rot_cross_pt);

    for (int i = 0; i < 3; i++)
      rotatedPt[i] = pt[i] + rot_cross_pt[i];
  }
}

template<typename T>
void radial_distort(const T* const rad_params,
                    T *proj) {
  T rsq, L;
  rsq = sqsum(2, proj);
  L = 1. + rad_params[0] * rsq + rad_params[1] * rsq * rsq;
  proj[0] = proj[0] * L;
  proj[1] = proj[1] * L;
}

template<typename T>
void project(const T* const cam,
             const T* const X,
             T* proj) {
  const T* const C = &cam[3];
  T Xo[3], Xcam[3];

  Xo[0] = X[0] - C[0];
  Xo[1] = X[1] - C[1];
  Xo[2] = X[2] - C[2];

  rodrigues_rotate_point(&cam[0], Xo, Xcam);

  proj[0] = Xcam[0] / Xcam[2];
  proj[1] = Xcam[1] / Xcam[2];

  radial_distort(&cam[9], proj);

  proj[0] = proj[0] * cam[6] + cam[7];
  proj[1] = proj[1] * cam[6] + cam[8];
}

template<typename T>
void computeReprojError(const T* const cam,
                        const T* const X,
                        const T* const w,
                        const double* const feat,
                        T *err) {
  T proj[2];
  project(cam, X, proj);

  err[0] = (*w)*(proj[0] - feat[0]);
  err[1] = (*w)*(proj[1] - feat[1]);
}

template<typename T>
void computeZachWeightError(const T* const w, T* err) {
  *err = 1 - (*w)*(*w);
}

template<typename T>
void objective(int n, int m, int p,
               const T* const cams,
               const T* const X,
               const T* const w,
               const int* const obs,
               const double* const feats,
               T* reproj_err,
               T* w_err) {
  for (int i = 0; i < p; i++) {
    int camIdx = obs[i * 2 + 0];
    int ptIdx = obs[i * 2 + 1];
    computeReprojError(&cams[camIdx * BA_NCAMPARAMS], &X[ptIdx * 3],
                       &w[i], &feats[i * 2], &reproj_err[2 * i]);
  }

  for (int i = 0; i < p; i++) {
    computeZachWeightError(&w[i], &w_err[i]);
  }
}

// rows is nrows+1 vector containing
// indices to cols and vals.
// rows[i] ... rows[i+1]-1 are elements of i-th row
// i.e. cols[row[i]] is the column of the first
// element in the row. Similarly for values.
class SparseMat {
public:
  int n, m, p; // number of cams, points and observations
  int nrows, ncols;
  std::vector<int> rows;
  std::vector<int> cols;
  std::vector<double> vals;

  SparseMat();
  SparseMat(int n_, int m_, int p_);

  void insert_reproj_err_block(int obsIdx,
                               int camIdx, int ptIdx, const double* const J);

  void insert_w_err_block(int wIdx, double w_d);

  void clear();
};

SparseMat::SparseMat() {}

SparseMat::SparseMat(int n_, int m_, int p_) : n(n_), m(m_), p(p_) {
  nrows = 2 * p + p;
  ncols = BA_NCAMPARAMS * n + 3 * m + p;
  rows.reserve(nrows + 1);
  int nnonzero = (BA_NCAMPARAMS + 3 + 1) * 2 * p + p;
  cols.reserve(nnonzero);
  vals.reserve(nnonzero);
  rows.push_back(0);
}

void SparseMat::insert_reproj_err_block(int obsIdx,
                                          int camIdx, int ptIdx, const double* const J) {
  int n_new_cols = BA_NCAMPARAMS + 3 + 1;
  rows.push_back(rows.back() + n_new_cols);
  rows.push_back(rows.back() + n_new_cols);

  for (int i_row = 0; i_row < 2; i_row++) {
      for (int i = 0; i < BA_NCAMPARAMS; i++) {
        cols.push_back(BA_NCAMPARAMS * camIdx + i);
        vals.push_back(J[2 * i + i_row]);
      }
      int col_offset = BA_NCAMPARAMS * n;
      int val_offset = BA_NCAMPARAMS * 2;
      for (int i = 0; i < 3; i++) {
        cols.push_back(col_offset + 3 * ptIdx + i);
        vals.push_back(J[val_offset + 2 * i + i_row]);
      }
      col_offset += 3 * m;
      val_offset += 3 * 2;
      cols.push_back(col_offset + obsIdx);
      vals.push_back(J[val_offset + i_row]);
    }
}

void SparseMat::insert_w_err_block(int wIdx, double w_d) {
  rows.push_back(rows.back() + 1);
  cols.push_back(BA_NCAMPARAMS * n + 3 * m + wIdx);
  vals.push_back(w_d);
}

void SparseMat::clear() {
  rows.clear();
  cols.clear();
  vals.clear();
  rows.reserve(nrows + 1);
  int nnonzero = (BA_NCAMPARAMS + 3 + 1) * 2 * p + p;
  cols.reserve(nnonzero);
  vals.reserve(nnonzero);
  rows.push_back(0);
}

struct Input {
  int n = 0, m = 0, p = 0;
  std::vector<double> cams, X, w, feats;
  std::vector<int> obs;
};

struct ObjOutput {
  std::vector<double> reproj_err;
  std::vector<double> w_err;
};

typedef SparseMat JacOutput;

using json = nlohmann::json;

void from_json(const json& j, Input& p) {
  p.n = j["n"].get<int>();
  p.m = j["m"].get<int>();
  p.p = j["p"].get<int>();

  auto cam = j["cam"].get<std::vector<double>>();
  auto x = j["x"].get<std::vector<double>>();
  auto w = j["w"].get<double>();
  auto feat = j["feat"].get<std::vector<double>>();

  int nCamParams = 11;

  p.cams.resize(nCamParams * p.n);
  p.X.resize(3 * p.m);
  p.w.resize(p.p);
  p.obs.resize(2 * p.p);
  p.feats.resize(2 * p.p);

  for (int i = 0; i < p.n; i++) {
    for (int j = 0; j < nCamParams; j++) {
      p.cams[i * nCamParams + j] = cam[j];
    }
  }

  for (int i = 0; i < p.m; i++) {
    for (int j = 0; j < 3; j++) {
      p.X[i*3+j] = x[j];
    }
  }

  for (int i = 0; i < p.p; i++) {
    p.w[i] = w;
  }

  int camIdx = 0;
  int ptIdx = 0;
  for (int i = 0; i < p.p; i++) {
    p.obs[i * 2 + 0] = (camIdx++ % p.n);
    p.obs[i * 2 + 1] = (ptIdx++ % p.m);
  }

  for (int i = 0; i < p.p; i++) {
    p.feats[i * 2 + 0] = feat[0];
    p.feats[i * 2 + 1] = feat[1];
  }
}

void to_json(nlohmann::json& j, const ObjOutput& p) {
  std::vector<double> reproj_err(2);
  reproj_err[0] = p.reproj_err[0];
  reproj_err[1] = p.reproj_err[1];
  j = {
    {"reproj_error",
     {{"elements", reproj_err},
      {"repeated", p.reproj_err.size()/2}}},
    {"w_err",
     {{"element", p.w_err[0]},
      {"repeated", p.w_err.size()}
     }
    }
  };
}

void to_json(nlohmann::json& j, const JacOutput& p) {
  j = {
    {"rows", p.rows},
    {"cols", p.cols},
    {"vals", p.vals}
  };
}

class Objective : public Function<Input, ObjOutput> {
public:
  Objective(ba::Input& input) : Function(input) {}

  void compute(ba::ObjOutput& output) {
    output.reproj_err.resize(2 * _input.p);
    output.w_err.resize(_input.p);
    ba::objective(_input.n, _input.m, _input.p,
                  _input.cams.data(), _input.X.data(), _input.w.data(),
                  _input.obs.data(), _input.feats.data(),
                  output.reproj_err.data(), output.w_err.data());
  }
};
}
