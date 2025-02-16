// Functions for converting between ADBench objects and the JSON
// format used by the Gradbench protocol.

#include "adbench/shared/HTData.h"

void read_HandInput_json(const char* fname, HandInput &input, int *runs);
void write_HandOutput_objective_json(std::ostream& f, HandOutput &output);
void write_HandOutput_jacobian_json(std::ostream& f, HandOutput &output);
