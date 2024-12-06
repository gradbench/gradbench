#include "ManualHT.h"
#include "adbench/main.h"
#include "adbench/shared/HTData.h"

int main(int argc, char* argv[]) {
  return generic_main<HandInput,
                      ManualHand,
                      read_HandInput_json,
                      write_HandOutput_objective_json,
                      write_HandOutput_jacobian_json>(argc, argv);
}
