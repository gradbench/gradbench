// Functions for converting between ADBench objects and the JSON
// format used by the Gradbench protocol.

#include "adbench/shared/GMMData.h"
#include "adbench/shared/BAData.h"
#include "adbench/shared/LSTMData.h"
#include "adbench/shared/HTData.h"

void read_GMMInput_json(const char* fname, GMMInput &input);
void write_GMMOutput_objective_json(std::ostream& f, GMMOutput &output);
void write_GMMOutput_jacobian_json(std::ostream& f, GMMOutput &output);

void read_BAInput_json(const char* fname, BAInput &input);
void write_BAOutput_objective_json(std::ostream& f, BAOutput &output);
void write_BAOutput_jacobian_json(std::ostream& f, BAOutput &output);

void read_LSTMInput_json(const char* fname, LSTMInput &input);
void write_LSTMOutput_objective_json(std::ostream& f, LSTMOutput &output);
void write_LSTMOutput_jacobian_json(std::ostream& f, LSTMOutput &output);

void read_HandInput_json(const char* fname, HandInput &input);
void write_HandOutput_objective_json(std::ostream& f, HandOutput &output);
void write_HandOutput_jacobian_json(std::ostream& f, HandOutput &output);
