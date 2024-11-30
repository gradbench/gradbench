#include <stdlib.h>
#include <iostream>
#include "TapenadeGMM.h"
#include "adbench/io.h"
#include "adbench/shared/GMMData.h"
#include "adbench/shared/utils.h"
#include "json.hpp"

int main(int argc, char* argv[]) {
  if (argc != 3 ||
      (std::string(argv[2]) != "F" && std::string(argv[2]) != "J")) {
    std::cerr << "Usage: " << argv[0] << " FILE <F|J>" << std::endl;
    exit(1);
  }

  const char* input_file = argv[1];

  GMMInput input;

  read_GMMInput_json(input_file, input);

  TapenadeGMM gmm;

  gmm.prepare(std::move(input));

  struct timespec start, finish;

  if (std::string(argv[2]) == "F") {
    clock_gettime( CLOCK_REALTIME, &start );
    gmm.calculate_objective(1);
    clock_gettime( CLOCK_REALTIME, &finish );

    GMMOutput output = gmm.output();
    write_GMMOutput_objective_json(std::cout, output);
  } else {
    clock_gettime( CLOCK_REALTIME, &start );
    gmm.calculate_jacobian(1);
    clock_gettime( CLOCK_REALTIME, &finish );

    GMMOutput output = gmm.output();
    write_GMMOutput_jacobian_json(std::cout, output);
  }
  std::cout << std::endl;

  double time_taken = (double) ((finish.tv_sec*1e9 + finish.tv_nsec) -
                                (start.tv_sec*1e9 + start.tv_nsec));
  std::cout << (long)time_taken;
}
