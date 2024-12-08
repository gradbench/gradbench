#include "EnzymeHello.h"
#include "adbench/main.h"
#include "adbench/shared/HelloData.h"

int main(int argc, char* argv[]) {
  return generic_main<HelloInput,
                      EnzymeHello,
                      read_HelloInput_json,
                      write_HelloOutput_objective_json,
                      write_HelloOutput_jacobian_json>(argc, argv);
}
