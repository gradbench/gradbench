#include "AdolCBA.h"
#include "adbench/main.h"
#include "adbench/shared/BAData.h"

int main(int argc, char* argv[]) {
  return generic_main<BAInput,
                      AdolCBA,
                      read_BAInput_json,
                      write_BAOutput_objective_json,
                      write_BAOutput_jacobian_json>(argc, argv);
}
