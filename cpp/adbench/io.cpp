#include "json.hpp"
#include "adbench/io.h"

void to_light_matrix(LightMatrix<double> &m,
                     std::vector<std::vector<double>> v) {
  // LightMatrixes are column-major for some unknown reason.
  m.resize(v[0].size(), v.size());
  for (size_t i = 0; i < v.size(); i++) {
    for (size_t j = 0; j < v[0].size(); j++) {
      m(j,i) = v[i][j];
    }
  }
}

void read_HandInput_json(const char* fname, HandInput &input, int *runs) {
  // Based on read_hand_instance from ADBench.
  using json = nlohmann::json;
  std::ifstream f(fname);
  json data = json::parse(f);

  input.theta = data["theta"].get<std::vector<double>>();

  auto us = data["us"].get<std::vector<std::vector<double>>>();
  input.us.resize(us.size()*2);
  for (size_t i = 0; i < us.size(); i++) {
    input.us[i*2+0] = us[i][0];
    input.us[i*2+1] = us[i][1];
  }

  input.data.correspondences =
    data["data"]["correspondences"].get<std::vector<int>>();

  to_light_matrix(input.data.points,
                  data["data"]["points"].get<std::vector<std::vector<double>>>());

  input.data.model.bone_names =
    data["data"]["model"]["bone_names"].get<std::vector<std::string>>();
  input.data.model.parents =
    data["data"]["model"]["parents"].get<std::vector<int>>();
  input.data.model.is_mirrored =
    data["data"]["model"]["is_mirrored"].get<bool>();

  to_light_matrix(input.data.model.base_positions,
                  data["data"]["model"]["base_positions"]
                  .get<std::vector<std::vector<double>>>());

  to_light_matrix(input.data.model.weights,
                  data["data"]["model"]["weights"]
                  .get<std::vector<std::vector<double>>>());

  auto triangles =
    data["data"]["model"]["triangles"].get<std::vector<std::vector<int>>>();
  for (auto t : triangles) {
    Triangle t2;
    t2.verts[0] = t[0];
    t2.verts[1] = t[1];
    t2.verts[2] = t[2];
    input.data.model.triangles.push_back(t2);
  }

  auto base_relatives =
    data["data"]["model"]["base_relatives"]
    .get<std::vector<std::vector<std::vector<double>>>>();
  input.data.model.base_relatives.resize(base_relatives.size());
  for (size_t i = 0; i < base_relatives.size(); i++) {
    to_light_matrix(input.data.model.base_relatives[i],
                    base_relatives[i]);
    input.data.model.base_relatives[i].transpose_in_place();
  }

  auto inverse_base_absolutes =
    data["data"]["model"]["inverse_base_absolutes"]
    .get<std::vector<std::vector<std::vector<double>>>>();
  input.data.model.inverse_base_absolutes.resize(inverse_base_absolutes.size());
  for (size_t i = 0; i < inverse_base_absolutes.size(); i++) {
    to_light_matrix(input.data.model.inverse_base_absolutes[i],
                    inverse_base_absolutes[i]);
    input.data.model.inverse_base_absolutes[i].transpose_in_place();
  }

  *runs = data["runs"];
}

void write_HandOutput_objective_json(std::ostream& f, HandOutput &output) {
  using json = nlohmann::json;
  f << json(output.objective);
}

void write_HandOutput_jacobian_json(std::ostream& f, HandOutput &output) {
  using json = nlohmann::json;
  int nrows = output.jacobian_nrows, ncols = output.jacobian_ncols;
  std::vector<std::vector<double>> out(nrows);
  for (int i = 0; i < nrows; i++) {
    out[i].resize(ncols);
    for (int j = 0; j < ncols; j++) {
      out[i][j] = output.jacobian[j*nrows+i];
    }
  }
  f << json(out);
}
