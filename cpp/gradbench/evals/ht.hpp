// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// Based on
// https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/cpp/shared/HandData.h
// https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/cpp/shared/hand_light_matrix.h

#pragma once

#include "adbench/shared/defs.h"
#include "adbench/shared/light_matrix.h"
#include "adbench/shared/matrix.h"
#include "gradbench/main.hpp"

namespace ht {

class ModelLightMatrix {
public:
  std::vector<std::string> bone_names;
  std::vector<int>
      parents;  // assumimng that parent is earlier in the order of bones
  std::vector<LightMatrix<double>> base_relatives;
  std::vector<LightMatrix<double>> inverse_base_absolutes;
  LightMatrix<double>              base_positions;
  LightMatrix<double>              weights;
  std::vector<Triangle>            triangles;
  bool                             is_mirrored;
};

class DataLightMatrix {
public:
  ModelLightMatrix    model;
  std::vector<int>    correspondences;
  LightMatrix<double> points;
};

using std::string;
using std::vector;

////////////////////////////////////////////////////////////
//////////////////// Declarations //////////////////////////
////////////////////////////////////////////////////////////

// theta: 26 [global rotation, global translation, finger parameters (4*5)]
// data: data measurements and hand model
// err: 3*number_of_correspondences
template <typename T>
void objective(const T* const theta, const DataLightMatrix& data, T* err);

// theta: 26 [global rotation, global translation, finger parameters (4*5)]
// us: 2*number_of_correspondences
// data: data measurements and hand model
// err: 3*number_of_correspondences
template <typename T>
void objective(const T* const theta, const T* const us,
               const DataLightMatrix& data, T* err);

////////////////////////////////////////////////////////////
//////////////////// Definitions ///////////////////////////
////////////////////////////////////////////////////////////

template <typename T>
void angle_axis_to_rotation_matrix(const T* const  angle_axis,
                                   LightMatrix<T>* pR) {
  T norm = sqrt(sqnorm(3, angle_axis));
  if (norm < .0001) {
    pR->set_identity();
    return;
  }

  T x = angle_axis[0] / norm;
  T y = angle_axis[1] / norm;
  T z = angle_axis[2] / norm;

  T s = sin(norm);
  T c = cos(norm);

  auto& R = *pR;
  R(0, 0) = x * x + (1 - x * x) * c;
  R(0, 1) = x * y * (1 - c) - z * s;
  R(0, 2) = x * z * (1 - c) + y * s;
  R(1, 0) = x * y * (1 - c) + z * s;
  R(1, 1) = y * y + (1 - y * y) * c;
  R(1, 2) = y * z * (1 - c) - x * s;
  R(2, 0) = x * z * (1 - c) - y * s;
  R(2, 1) = z * y * (1 - c) + x * s;
  R(2, 2) = z * z + (1 - z * z) * c;
}

template <typename T>
void apply_global_transform(const LightMatrix<T>& pose_params,
                            LightMatrix<T>*       positions) {
  LightMatrix<T> R(3, 3);
  angle_axis_to_rotation_matrix(pose_params.get_col(0), &R);

  for (int i = 0; i < 3; i++) {
    R.scale_col(i, pose_params(i, 1));
  }

  LightMatrix<T> tmp;
  mat_mult(R, *positions, &tmp);
#pragma omp parallel for collapse(2)
  for (int j = 0; j < positions->ncols_; j++)
    for (int i = 0; i < positions->nrows_; i++)
      (*positions)(i, j) = tmp(i, j) + pose_params(i, 2);
}

template <typename T>
void relatives_to_absolutes(const vector<LightMatrix<T>>& relatives,
                            const vector<int>&            parents,
                            vector<LightMatrix<T>>*       pabsolutes) {
  auto& absolutes = *pabsolutes;
  absolutes.resize(parents.size());
  for (size_t i = 0; i < parents.size(); i++) {
    if (parents[i] == -1)
      absolutes[i] = relatives[i];
    else
      mat_mult(absolutes[parents[i]], relatives[i], &absolutes[i]);
  }
}

template <typename T>
void euler_angles_to_rotation_matrix(const T* const xzy, LightMatrix<T>* pR) {
  T              tx = xzy[0], ty = xzy[2], tz = xzy[1];
  LightMatrix<T> Rx(3, 3), Ry(3, 3), Rz(3, 3);
  Rx.set_identity();
  Rx(1, 1) = cos(tx);
  Rx(2, 1) = sin(tx);
  Rx(1, 2) = -Rx(2, 1);
  Rx(2, 2) = Rx(1, 1);

  Ry.set_identity();
  Ry(0, 0) = cos(ty);
  Ry(0, 2) = sin(ty);
  Ry(2, 0) = -Ry(0, 2);
  Ry(2, 2) = Ry(0, 0);

  Rz.set_identity();
  Rz(0, 0) = cos(tz);
  Rz(1, 0) = sin(tz);
  Rz(0, 1) = -Rz(1, 0);
  Rz(1, 1) = Rz(0, 0);

  LightMatrix<T> tmp;
  mat_mult(Rz, Ry, &tmp);
  mat_mult(tmp, Rx, pR);
}

template <typename T>
void get_posed_relatives(const ModelLightMatrix& model,
                         const LightMatrix<T>&   pose_params,
                         vector<LightMatrix<T>>* prelatives) {
  auto& relatives = *prelatives;
  relatives.resize(model.base_relatives.size());

  int offset = 3;
  for (size_t i = 0; i < model.bone_names.size(); i++) {
    LightMatrix<T> tr(4, 4);
    tr.set_identity();

    LightMatrix<T> R(3, 3);
    euler_angles_to_rotation_matrix(pose_params.get_col((int)i + offset), &R);
    tr.set_block(0, 0, R);

    mat_mult(model.base_relatives[i], tr, &relatives[i]);
  }
}

template <typename T>
void get_skinned_vertex_positions(const ModelLightMatrix& model,
                                  const LightMatrix<T>&   pose_params,
                                  LightMatrix<T>*         ppositions,
                                  bool                    apply_global) {
  vector<LightMatrix<T>> relatives, absolutes, transforms;
  get_posed_relatives(model, pose_params, &relatives);
  relatives_to_absolutes(relatives, model.parents, &absolutes);

  // Get bone transforms.
  transforms.resize(absolutes.size());
#pragma omp parallel for
  for (size_t i = 0; i < absolutes.size(); i++) {
    mat_mult(absolutes[i], model.inverse_base_absolutes[i], &transforms[i]);
  }

  // Transform vertices by necessary transforms. + apply skinning
  auto& positions = *ppositions;
  positions.resize(3, model.base_positions.ncols_);
  positions.fill(0.);
  LightMatrix<T> curr_positions(4, model.base_positions.ncols_);
#pragma omp parallel for firstprivate(curr_positions)
  for (size_t i_bone = 0; i_bone < transforms.size(); i_bone++) {
    mat_mult(transforms[i_bone], model.base_positions, &curr_positions);
    for (int i_vert = 0; i_vert < positions.ncols_; i_vert++)
      for (int i = 0; i < 3; i++) {
        T change = curr_positions(i, i_vert) * model.weights((int)i_bone, i_vert);
#pragma omp atomic
        positions(i, i_vert) += change;
      }
  }

  if (model.is_mirrored)
    positions.scale_row(0, -1);

  if (apply_global)
    apply_global_transform(pose_params, &positions);
}

//% !!!!!!! fixed order pose_params !!!!!
//% 1) global_rotation 2) scale 3) global_translation
//% 4) wrist
//% 5) thumb1, 6)thumb2, 7) thumb3, 8) thumb4
//%       similarly: index, middle, ring, pinky
//%       end) forearm
template <typename T>
void to_pose_params(const T* const theta, const vector<string>& bone_names,
                    LightMatrix<T>* ppose_params) {
  auto& pose_params = *ppose_params;
  pose_params.resize(3, (int)bone_names.size() + 3);
  pose_params.fill(0.);

  pose_params.set_col(0, &theta[0]);
  pose_params.set_col(1, 1.);
  pose_params.set_col(2, &theta[3]);

  int i_theta       = 6;
  int i_pose_params = 5;
  int n_fingers     = 5;
  for (int i_finger = 0; i_finger < n_fingers; i_finger++) {
    for (int i = 2; i <= 4; i++) {
      pose_params(0, i_pose_params) = theta[i_theta++];
      if (i == 2) {
        pose_params(1, i_pose_params) = theta[i_theta++];
      }
      i_pose_params++;
    }
    i_pose_params++;
  }
}

template <typename T>
void objective(const T* const theta, const DataLightMatrix* data, T* err) {
  LightMatrix<T> pose_params;
  to_pose_params(theta, data->model.bone_names, &pose_params);

  LightMatrix<T> vertex_positions;
  get_skinned_vertex_positions(data->model, pose_params, &vertex_positions,
                               true);

#pragma omp parallel for
  for (size_t i = 0; i < data->correspondences.size(); i++)
    for (int j = 0; j < 3; j++)
      err[i * 3 + j] =
          data->points(j, i) - vertex_positions(j, data->correspondences[i]);
}

template <typename T>
void objective(const T* const theta, const T* const us,
               const DataLightMatrix* data, T* err) {
  LightMatrix<T> pose_params;
  to_pose_params(theta, data->model.bone_names, &pose_params);

  LightMatrix<T> vertex_positions;
  get_skinned_vertex_positions(data->model, pose_params, &vertex_positions,
                               true);

#pragma omp parallel for
  for (size_t i = 0; i < data->correspondences.size(); i++) {
    const auto& verts = data->model.triangles[data->correspondences[i]].verts;
    const T* const u  = &us[2 * i];
    for (int j = 0; j < 3; j++) {
      T hand_point_coord = u[0] * vertex_positions(j, verts[0]) +
                           u[1] * vertex_positions(j, verts[1]) +
                           (1. - u[0] - u[1]) * vertex_positions(j, verts[2]);

      err[i * 3 + j] = data->points(j, i) - hand_point_coord;
    }
  }
}

struct Input {
  std::vector<double> theta;
  DataLightMatrix     data;
  std::vector<double> us;
};

typedef std::vector<double> ObjOutput;

struct JacOutput {
  int                 jacobian_ncols, jacobian_nrows;
  std::vector<double> jacobian;
};

using json = nlohmann::json;

void to_light_matrix(LightMatrix<double>&             m,
                     std::vector<std::vector<double>> v) {
  // LightMatrixes are column-major for some unknown reason.
  m.resize(v[0].size(), v.size());
  for (size_t i = 0; i < v.size(); i++) {
    for (size_t j = 0; j < v[0].size(); j++) {
      m(j, i) = v[i][j];
    }
  }
}

void from_json(const json& j, Input& p) {
  p.theta = j["theta"].get<std::vector<double>>();

  auto us = j["us"].get<std::vector<std::vector<double>>>();
  p.us.resize(us.size() * 2);
  for (size_t i = 0; i < us.size(); i++) {
    p.us[i * 2 + 0] = us[i][0];
    p.us[i * 2 + 1] = us[i][1];
  }

  p.data.correspondences = j["data"]["correspondences"].get<std::vector<int>>();

  to_light_matrix(p.data.points,
                  j["data"]["points"].get<std::vector<std::vector<double>>>());

  p.data.model.bone_names =
      j["data"]["model"]["bone_names"].get<std::vector<std::string>>();
  p.data.model.parents = j["data"]["model"]["parents"].get<std::vector<int>>();
  p.data.model.is_mirrored = j["data"]["model"]["is_mirrored"].get<bool>();

  to_light_matrix(p.data.model.base_positions,
                  j["data"]["model"]["base_positions"]
                      .get<std::vector<std::vector<double>>>());

  to_light_matrix(
      p.data.model.weights,
      j["data"]["model"]["weights"].get<std::vector<std::vector<double>>>());

  auto triangles =
      j["data"]["model"]["triangles"].get<std::vector<std::vector<int>>>();
  for (auto t : triangles) {
    Triangle t2;
    t2.verts[0] = t[0];
    t2.verts[1] = t[1];
    t2.verts[2] = t[2];
    p.data.model.triangles.push_back(t2);
  }

  auto base_relatives =
      j["data"]["model"]["base_relatives"]
          .get<std::vector<std::vector<std::vector<double>>>>();
  p.data.model.base_relatives.resize(base_relatives.size());
  for (size_t i = 0; i < base_relatives.size(); i++) {
    to_light_matrix(p.data.model.base_relatives[i], base_relatives[i]);
    p.data.model.base_relatives[i].transpose_in_place();
  }

  auto inverse_base_absolutes =
      j["data"]["model"]["inverse_base_absolutes"]
          .get<std::vector<std::vector<std::vector<double>>>>();
  p.data.model.inverse_base_absolutes.resize(inverse_base_absolutes.size());
  for (size_t i = 0; i < inverse_base_absolutes.size(); i++) {
    to_light_matrix(p.data.model.inverse_base_absolutes[i],
                    inverse_base_absolutes[i]);
    p.data.model.inverse_base_absolutes[i].transpose_in_place();
  }
}

void to_json(nlohmann::json& j, const JacOutput& p) {
  int nrows = p.jacobian_nrows, ncols = p.jacobian_ncols;
  std::vector<std::vector<double>> out(nrows);
  for (int i = 0; i < nrows; i++) {
    out[i].resize(ncols);
    for (int j = 0; j < ncols; j++) {
      out[i][j] = p.jacobian[j * nrows + i];
    }
  }
  j = out;
}

class Objective : public Function<Input, ObjOutput> {
  bool _complicated = false;

public:
  Objective(Input& input)
      : Function(input), _complicated(input.us.size() != 0) {}

  void compute(ObjOutput& output) {
    int err_size = 3 * _input.data.correspondences.size();
    output.resize(err_size);

    if (_complicated) {
      objective(_input.theta.data(), _input.us.data(), &_input.data,
                output.data());
    } else {
      objective(_input.theta.data(), &_input.data, output.data());
    }
  }
};

}  // namespace ht
