#include "EnzymeLSTM.h"
#include "adbench/main.h"
#include "adbench/shared/LSTMData.h"

int main(int argc, char* argv[]) {
  return generic_main<LSTMInput,
                      EnzymeLSTM,
                      read_LSTMInput_json,
                      write_LSTMOutput_objective_json,
                      write_LSTMOutput_jacobian_json>(argc, argv);
}
