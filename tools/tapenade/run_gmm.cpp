#include <stdlib.h>
#include <iostream>
#include "TapenadeGMM.h"
#include "adbench/shared/GMMData.h"
#include "adbench/shared/utils.h"
#include "json.hpp"

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

void write_GMMOutput_F_json(std::ostream& f, GMMOutput &output) {
  using json = nlohmann::json;
  f << json(output.objective);
}

void write_GMMOutput_J_json(std::ostream& f, GMMOutput &output) {
  using json = nlohmann::json;
  f << json(output.gradient);
}

int main(int argc, char* argv[]) {
  if (argc != 3 ||
      std::string(argv[2]) != "F"
      && std::string(argv[2]) != "J") {
    std::cerr << "Usage: " << argv[0] << " FILE <F|J>" << std::endl;
    exit(1);
  }

  const char* input_file = argv[1];

  GMMInput input;
  bool replicate_point = false;

  read_GMMInput_json(input_file, input);

  TapenadeGMM gmm;

  gmm.prepare(std::move(input));

  struct timespec start, finish;

  if (std::string(argv[2]) == "F") {
    clock_gettime( CLOCK_REALTIME, &start );
    gmm.calculate_objective(1);
    clock_gettime( CLOCK_REALTIME, &finish );

    GMMOutput output = gmm.output();
    write_GMMOutput_F_json(std::cout, output);
  } else {
    clock_gettime( CLOCK_REALTIME, &start );
    gmm.calculate_jacobian(1);
    clock_gettime( CLOCK_REALTIME, &finish );

    GMMOutput output = gmm.output();
    write_GMMOutput_J_json(std::cout, output);
  }
  std::cout << std::endl;

  double time_taken = (double) (finish.tv_nsec - start.tv_nsec);
  std::cout << (long)time_taken;
}
