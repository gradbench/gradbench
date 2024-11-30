#include "json.hpp"
#include "adbench/io.h"
#include "adbench/shared/GMMData.h"
#include "adbench/shared/BAData.h"

void read_GMMInput_json(const char* fname, GMMInput &input) {
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
  // This code based on read_ba_instance from ADBench.
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
        {"rows", output.J.rows.size() - 1},
        {"columns", output.J.cols[output.J.cols.size()-1] + 1}
      }}
  };
  f << out;
}
