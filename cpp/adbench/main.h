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

  int runs;
  Input input;

  ReadInput(input_file, input, &runs);

  Benchmark b;

  double prepare_time_taken;
  {
    struct timespec start, finish;
    clock_gettime( CLOCK_REALTIME, &start);
    b.prepare(std::move(input));
    clock_gettime( CLOCK_REALTIME, &finish);
    prepare_time_taken = (finish.tv_sec*1e9 + finish.tv_nsec) - (start.tv_sec*1e9 + start.tv_nsec);
  }

  struct timespec start[runs], finish[runs];

  if (std::string(argv[2]) == "F") {
    for (int i = 0; i < runs; i++) {
      clock_gettime( CLOCK_REALTIME, &start[i] );
      b.calculate_objective(1);
      clock_gettime( CLOCK_REALTIME, &finish[i] );
    }

    auto output = b.output();
    WriteObjective(std::cout, output);
  } else {
    for (int i = 0; i < runs; i++) {
      clock_gettime( CLOCK_REALTIME, &start[i] );
      b.calculate_jacobian(1);
      clock_gettime( CLOCK_REALTIME, &finish[i] );
    }

    auto output = b.output();
    WriteJacobian(std::cout, output);
  }
  std::cout << std::endl;

  std::cout << "{\"name\": \"prepare\", \"nanoseconds\": " << (long)prepare_time_taken << "}" << std::endl;
  for (int i = 0; i < runs; i++) {
    long time_taken = ((finish[i].tv_sec*1e9 + finish[i].tv_nsec) -
                       (start[i].tv_sec*1e9 + start[i].tv_nsec));
    std::cout << "{\"name\": \"evaluate\", \"nanoseconds\": " << (long)time_taken << "}" << std::endl;
  }

  return 0;
}
