#include <stdlib.h>
#include <iostream>
#include "TapenadeLSTM.h"
#include "adbench/io.h"
#include "adbench/shared/LSTMData.h"
#include "adbench/shared/utils.h"
#include "json.hpp"

int main(int argc, char* argv[]) {
  if (argc != 3 ||
      (std::string(argv[2]) != "F" && (std::string(argv[2]) != "J"))) {
    std::cerr << "Usage: " << argv[0] << " FILE <F|J>" << std::endl;
    exit(1);
  }

  const char* input_file = argv[1];

  LSTMInput input;

  read_LSTMInput_json(input_file, input);

  TapenadeLSTM lstm;

  lstm.prepare(std::move(input));

  struct timespec start, finish;

  if (std::string(argv[2]) == "F") {
    clock_gettime( CLOCK_REALTIME, &start );
    lstm.calculate_objective(1);
    clock_gettime( CLOCK_REALTIME, &finish );

    LSTMOutput output = lstm.output();
    write_LSTMOutput_objective_json(std::cout, output);
  } else {
    clock_gettime( CLOCK_REALTIME, &start );
    lstm.calculate_jacobian(1);
    clock_gettime( CLOCK_REALTIME, &finish );

    LSTMOutput output = lstm.output();
    write_LSTMOutput_jacobian_json(std::cout, output);
  }
  std::cout << std::endl;

  double time_taken = (double) ((finish.tv_sec*1e9 + finish.tv_nsec) -
                                (start.tv_sec*1e9 + start.tv_nsec));
  std::cout << (long)time_taken;
}
