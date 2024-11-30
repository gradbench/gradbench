#include <stdlib.h>
#include <iostream>
#include "TapenadeHT.h"
#include "adbench/io.h"
#include "adbench/shared/HTData.h"
#include "adbench/shared/utils.h"
#include "json.hpp"

int main(int argc, char* argv[]) {
  if (argc != 3 ||
      (std::string(argv[2]) != "F" && (std::string(argv[2]) != "J"))) {
    std::cerr << "Usage: " << argv[0] << " FILE <F|J>" << std::endl;
    exit(1);
  }

  const char* input_file = argv[1];

  HandInput input;

  read_HandInput_json(input_file, input);

  TapenadeHand ht;

  ht.prepare(std::move(input));

  struct timespec start, finish;

  if (std::string(argv[2]) == "F") {
    clock_gettime( CLOCK_REALTIME, &start );
    ht.calculate_objective(1);
    clock_gettime( CLOCK_REALTIME, &finish );

    HandOutput output = ht.output();
    write_HandOutput_objective_json(std::cout, output);
  } else {
    clock_gettime( CLOCK_REALTIME, &start );
    ht.calculate_jacobian(1);
    clock_gettime( CLOCK_REALTIME, &finish );

    HandOutput output = ht.output();
    write_HandOutput_jacobian_json(std::cout, output);
  }
  std::cout << std::endl;

  double time_taken = (double) (finish.tv_nsec - start.tv_nsec);
  std::cout << (long)time_taken;
}

