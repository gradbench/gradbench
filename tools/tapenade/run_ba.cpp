#include <stdlib.h>
#include <iostream>
#include "TapenadeBA.h"
#include "adbench/shared/BAData.h"
#include "adbench/shared/utils.h"
#include "json.hpp"

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

void write_BAOutput_F_json(std::ostream& f, BAOutput &output) {
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

void write_BAOutput_J_json(std::ostream& f, BAOutput &output) {
  using json = nlohmann::json;
  json out = {
    {"BASparseMat", {
        {"rows", output.J.rows.size() - 1},
        {"columns", output.J.cols[output.J.cols.size()-1] + 1}
      }}
  };
  f << out;
}

int main(int argc, char* argv[]) {
  if (argc != 3 ||
      std::string(argv[2]) != "F"
      && std::string(argv[2]) != "J") {
    std::cerr << "Usage: " << argv[0] << " FILE <F|J>" << std::endl;
    exit(1);
  }

  const char* input_file = argv[1];

  BAInput input;

  read_BAInput_json(input_file, input);

  TapenadeBA ba;

  ba.prepare(std::move(input));

  struct timespec start, finish;

  if (std::string(argv[2]) == "F") {
    clock_gettime( CLOCK_REALTIME, &start );
    ba.calculate_objective(1);
    clock_gettime( CLOCK_REALTIME, &finish );

    BAOutput output = ba.output();
    write_BAOutput_F_json(std::cout, output);
  } else {
    clock_gettime( CLOCK_REALTIME, &start );
    ba.calculate_jacobian(1);
    clock_gettime( CLOCK_REALTIME, &finish );

    BAOutput output = ba.output();
    write_BAOutput_J_json(std::cout, output);
  }
  std::cout << std::endl;

  double time_taken = (double) (finish.tv_nsec - start.tv_nsec);
  std::cout << (long)time_taken;
}
