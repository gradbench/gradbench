// A generic main function for tools built on C++.

#pragma once

#include <fstream>
#include <map>

// To implement a function for an eval, define a class that inherits
// from this one. The InputP type must be deserialisable from JSON
// (with the to_json overloaded function), and OutputP must be
// serialisable to JSON (with from_json).
template <typename InputP, typename OutputP>
class Function {
protected:
  InputP& _input;
public:
  typedef InputP Input;
  typedef OutputP Output;
  Function(Input& input) : _input(input) {}
  virtual ~Function() = default;
  virtual void compute(Output&) = 0;
};

template<typename Benchmark>
int function_main(const std::string& input_file) {
  using json = nlohmann::json;
  std::ifstream f(input_file);
  json j = json::parse(f);
  typename Benchmark::Input input = j.template get<typename Benchmark::Input>();
  int runs = j["runs"];
  assert(runs > 0);

  double prepare_time_taken;
  struct timespec prepare_start, prepare_finish;

  clock_gettime( CLOCK_REALTIME, &prepare_start);
  Benchmark b(input);
  clock_gettime( CLOCK_REALTIME, &prepare_finish);
  prepare_time_taken =
    (prepare_finish.tv_sec*1e9 + prepare_finish.tv_nsec) -
    (prepare_start.tv_sec*1e9 + prepare_start.tv_nsec);

  std::vector<struct timespec> start(runs), finish(runs);

  typename Benchmark::Output output;

  for (int i = 0; i < runs; i++) {
    clock_gettime( CLOCK_REALTIME, &start[i] );
    b.compute(output);
    clock_gettime( CLOCK_REALTIME, &finish[i] );
  }

  std::cout << json(output) << std::endl;

  std::cout << "{\"name\": \"prepare\", \"nanoseconds\": " << (long)prepare_time_taken << "}" << std::endl;

  for (int i = 0; i < runs; i++) {
    long time_taken = ((finish[i].tv_sec*1e9 + finish[i].tv_nsec) -
                       (start[i].tv_sec*1e9 + start[i].tv_nsec));
    std::cout << "{\"name\": \"evaluate\", \"nanoseconds\": " << (long)time_taken << "}" << std::endl;
  }

  return 0;
}

int generic_main(int argc, char* argv[],
                 std::map<std::string, int (*)(const std::string&)> &&m) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " FILE FUNCTION" << std::endl;
    exit(1);
  }

  std::string input_file = argv[1];
  std::string function = argv[2];

  for (auto it : m) {
    if (it.first == function) {
      return it.second(input_file);
    }
  }

  std::cerr << "Unknown function: " << function << std::endl;
  return 1;
}
