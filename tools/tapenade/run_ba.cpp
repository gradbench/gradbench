#include <stdlib.h>
#include <iostream>
#include "TapenadeBA.h"
#include "adbench/io.h"
#include "adbench/shared/BAData.h"
#include "adbench/shared/utils.h"
#include "json.hpp"

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
    write_BAOutput_objective_json(std::cout, output);
  } else {
    clock_gettime( CLOCK_REALTIME, &start );
    ba.calculate_jacobian(1);
    clock_gettime( CLOCK_REALTIME, &finish );

    BAOutput output = ba.output();
    write_BAOutput_jacobian_json(std::cout, output);
  }
  std::cout << std::endl;

  double time_taken = (double) (finish.tv_nsec - start.tv_nsec);
  std::cout << (long)time_taken;
}
