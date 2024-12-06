// Generic main function for use with C++ implementations of ADBench
// benchmarks that use ADBench data structures for representing
// input/output.

#include <stdlib.h>
#include <iostream>
#include "adbench/io.h"
#include "adbench/shared/utils.h"
#include "json.hpp"

template<typename Input, typename Benchmark, auto ReadInput, auto WriteObjective, auto WriteJacobian>
int generic_main(int argc, char* argv[]) {
  if (argc != 3 ||
      (std::string(argv[2]) != "F" && (std::string(argv[2]) != "J"))) {
    std::cerr << "Usage: " << argv[0] << " FILE <F|J>" << std::endl;
    exit(1);
  }

  const char* input_file = argv[1];

  Input input;

  ReadInput(input_file, input);

  Benchmark b;

  b.prepare(std::move(input));

  struct timespec start, finish;

  if (std::string(argv[2]) == "F") {
    clock_gettime( CLOCK_REALTIME, &start );
    b.calculate_objective(1);
    clock_gettime( CLOCK_REALTIME, &finish );

    auto output = b.output();
    WriteObjective(std::cout, output);
  } else {
    clock_gettime( CLOCK_REALTIME, &start );
    b.calculate_jacobian(1);
    clock_gettime( CLOCK_REALTIME, &finish );

    auto output = b.output();
    WriteJacobian(std::cout, output);
  }
  std::cout << std::endl;

  double time_taken = (double) ((finish.tv_sec*1e9 + finish.tv_nsec) -
                                (start.tv_sec*1e9 + start.tv_nsec));
  std::cout << (long)time_taken;

  return 0;
}
