#include "ManualKMeans.h"
#include "adbench/main.h"
#include "adbench/shared/KMeansData.h"

int main(int argc, char* argv[]) {
  return generic_main<KMeansInput,
                      ManualKMeans,
                      read_KMeansInput_json,
                      write_KMeansOutput_objective_json,
                      write_KMeansOutput_jacobian_json>(argc, argv);
}
