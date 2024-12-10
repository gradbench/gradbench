#include "json.hpp"
#include "adbench/io.h"
#include "adbench/shared/GMMData.h"
#include "adbench/shared/BAData.h"

void read_HelloInput_json(const char* fname, HelloInput &input) {
  using json = nlohmann::json;
  std::ifstream f(fname);
  json data = json::parse(f);
  input.x = data.get<double>();
}

void write_HelloOutput_objective_json(std::ostream& f, HelloOutput &output) {
  using json = nlohmann::json;
  f << json(output.objective);
}

void write_HelloOutput_jacobian_json(std::ostream& f, HelloOutput &output) {
  using json = nlohmann::json;
  f << json(output.gradient);
}

void read_GMMInput_json(const char* fname, GMMInput &input) {
  // Based on read_lstm_instance from ADBench.
  using json = nlohmann::json;
  std::ifstream f(fname);
  json data = json::parse(f);
  input.d = data["d"].get<int>();
  input.k = data["k"].get<int>();
  input.n = data["n"].get<int>();
  input.alphas = data["alpha"].get<std::vector<double>>();

  auto means = data["means"].get<std::vector<std::vector<double>>>();
  auto icf = data["icf"].get<std::vector<std::vector<double>>>();
  auto x = data["x"].get<std::vector<std::vector<double>>>();
  for (int i = 0; i < input.k; i++) {
    input.means.insert(input.means.end(), means[i].begin(), means[i].end());
    input.icf.insert(input.icf.end(), icf[i].begin(), icf[i].end());
  }
  for (int i = 0; i < input.n; i++) {
    input.x.insert(input.x.end(), x[i].begin(), x[i].end());
  }

  input.wishart.gamma = data["gamma"].get<double>();
  input.wishart.m = data["m"].get<int>();
}

void write_GMMOutput_objective_json(std::ostream& f, GMMOutput &output) {
  using json = nlohmann::json;
  f << json(output.objective);
}

void write_GMMOutput_jacobian_json(std::ostream& f, GMMOutput &output) {
  using json = nlohmann::json;
  f << json(output.gradient);
}


void read_BAInput_json(const char* fname, BAInput &input) {
  // Based on read_ba_instance from ADBench.
  using json = nlohmann::json;
  std::ifstream f(fname);
  json data = json::parse(f);
  input.n = data["n"].get<int>();
  input.m = data["m"].get<int>();
  input.p = data["p"].get<int>();

  auto cam = data["cam"].get<std::vector<double>>();
  auto x = data["x"].get<std::vector<double>>();
  auto w = data["w"].get<double>();
  auto feat = data["feat"].get<std::vector<double>>();

  int nCamParams = 11;

  input.cams.resize(nCamParams * input.n);
  input.X.resize(3 * input.m);
  input.w.resize(input.p);
  input.obs.resize(2 * input.p);
  input.feats.resize(2 * input.p);

  for (int i = 0; i < input.n; i++) {
    for (int j = 0; j < nCamParams; j++) {
      input.cams[i * nCamParams + j] = cam[j];
    }
  }

  for (int i = 0; i < input.m; i++) {
    for (int j = 0; j < 3; j++) {
      input.X[i*3+j] = x[j];
    }
  }

  for (int i = 0; i < input.p; i++) {
    input.w[i] = w;
  }

  int camIdx = 0;
  int ptIdx = 0;
  for (int i = 0; i < input.p; i++) {
    input.obs[i * 2 + 0] = (camIdx++ % input.n);
    input.obs[i * 2 + 1] = (ptIdx++ % input.m);
  }

  for (int i = 0; i < input.p; i++) {
    input.feats[i * 2 + 0] = feat[0];
    input.feats[i * 2 + 1] = feat[1];
  }
}

void write_BAOutput_objective_json(std::ostream& f, BAOutput &output) {
  using json = nlohmann::json;
  std::vector<double> reproj_err(2);
  reproj_err[0] = output.reproj_err[0];
  reproj_err[1] = output.reproj_err[1];
  json out = {
    {"reproj_error",
     {{"elements", reproj_err},
      {"repeated", output.reproj_err.size()/2}}},
    {"w_err",
     {{"element", output.w_err[0]},
      {"repeated", output.w_err.size()}
     }
    }
  };
  f << out;
}

void write_BAOutput_jacobian_json(std::ostream& f, BAOutput &output) {
  using json = nlohmann::json;
  json out = {
    {"BASparseMat", {
        {"rows", output.J.rows},
        {"cols", output.J.cols},
        {"vals", output.J.vals}
      }}
  };
  f << out;
}

void read_LSTMInput_json(const char* fname, LSTMInput &input) {
  // Based on read_lstm_instance from ADBench.
  using json = nlohmann::json;
  std::ifstream f(fname);
  json data = json::parse(f);

  auto main_params = data["main_params"].get<std::vector<std::vector<double>>>();
  auto extra_params = data["extra_params"].get<std::vector<std::vector<double>>>();
  auto state = data["state"].get<std::vector<std::vector<double>>>();
  auto sequence = data["sequence"].get<std::vector<std::vector<double>>>();

  input.l = main_params.size() / 2;
  input.b = main_params[0].size() / 4;
  input.c = sequence.size();

  for (auto it = main_params.begin(); it != main_params.end(); it++) {
    input.main_params.insert(input.main_params.end(), it->begin(), it->end());
  }

  for (auto it = extra_params.begin(); it != extra_params.end(); it++) {
    input.extra_params.insert(input.extra_params.end(), it->begin(), it->end());
  }

  for (auto it = state.begin(); it != state.end(); it++) {
    input.state.insert(input.state.end(), it->begin(), it->end());
  }

  for (auto it = sequence.begin(); it != sequence.end(); it++) {
    input.sequence.insert(input.sequence.end(), it->begin(), it->end());
  }
}

void write_LSTMOutput_objective_json(std::ostream& f, LSTMOutput &output) {
  using json = nlohmann::json;
  f << json(output.objective);
}

void write_LSTMOutput_jacobian_json(std::ostream& f, LSTMOutput &output) {
  using json = nlohmann::json;
  f << json(output.gradient);
}

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

void read_HandInput_json(const char* fname, HandInput &input) {
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
