#include <array>
#include <cmath>

#include "ad.hpp"
#include "gradbench/evals/saddle.hpp"
#include "gradbench/gd.hpp"
#include "gradbench/main.hpp"

template <typename T>
class argmax_y_tangent_driver {
  using active_t = ad::tangent_t<T>;

  std::array<T, 2> _optimizer_start;
  mutable T        _x1, _x2;  // set when calling compute_the_argmax

public:
  argmax_y_tangent_driver(T const* start) {
    _optimizer_start[0] = start[0];
    _optimizer_start[1] = start[1];
  }

  constexpr size_t input_size() const { return 2; }

  void capture(T x1, T x2) const {
    _x1 = x1;
    _x2 = x2;
  }

  void objective(T const* in, T* out) const {
    *out = saddle::objective(_x1, _x2, in[0], in[1]);
  }

  void gradient(T const* in, T* out) const {
    active_t x1_active, x2_active, y1_active, y2_active, res;
    x1_active = _x1;
    x2_active = _x2;
    y1_active = in[0];
    y2_active = in[1];

    ad::derivative(y1_active) = 1.0;
    ad::derivative(y2_active) = 0.0;
    res    = saddle::objective(x1_active, x2_active, y1_active, y2_active);
    out[0] = ad::derivative(res);

    ad::derivative(y1_active) = 0.0;
    ad::derivative(y2_active) = 1.0;
    res    = saddle::objective(x1_active, x2_active, y1_active, y2_active);
    out[1] = ad::derivative(res);
  }

  std::vector<T> compute_the_argmax() const {
    return multivariate_argmax(*this, _optimizer_start.data());
  }
};

template <typename T>
class argmax_y_adjoint_driver {
  using adjoint  = ad::adjoint<T>;
  using active_t = ad::adjoint_t<T>;

  ad::shared_global_tape_ptr<adjoint> _tape;
  std::array<T, 2>                    _optimizer_start;
  mutable T _x1, _x2;  // set when calling compute_the_argmax

public:
  argmax_y_adjoint_driver(T const* start) {
    _optimizer_start[0] = start[0];
    _optimizer_start[1] = start[1];
  }

  constexpr size_t input_size() const { return 2; }

  void capture(T x1, T x2) const {
    _x1 = x1;
    _x2 = x2;
  }

  void objective(T const* in, T* out) const {
    *out = saddle::objective(_x1, _x2, in[0], in[1]);
  }

  void gradient(T const* in, T* out) const {
    active_t x1_active, x2_active, y1_active, y2_active, res;
    x1_active = _x1;
    x2_active = _x2;
    y1_active = in[0];
    y2_active = in[1];

    _tape->reset();
    _tape->register_variable(y1_active);
    _tape->register_variable(y2_active);
    res = saddle::objective(x1_active, x2_active, y1_active, y2_active);
    ad::derivative(res) = 1.0;
    _tape->interpret_adjoint();
    out[0] = ad::derivative(y1_active);
    out[1] = ad::derivative(y2_active);
  }

  std::vector<T> compute_the_argmax() const {
    return multivariate_argmax(*this, _optimizer_start.data());
  }
};

template <template <typename> typename INNER_DRIVER_T>
struct argmin_x_tangent_driver {
  using active_t = ad::tangent_t<double>;
  INNER_DRIVER_T<double> driver;

  std::array<double, 2> _start;

  argmin_x_tangent_driver(double const* start) : driver(start), _start() {
    _start[0] = start[0];
    _start[1] = start[1];
  }

  constexpr size_t input_size() const { return 2; }

  void objective(double const* in, double* out) const {
    driver.capture(in[0], in[1]);
    std::vector<double> y_cur = driver.compute_the_argmax();
    *out = saddle::objective(in[0], in[1], y_cur[0], y_cur[1]);
  }

  void gradient(double const* in, double* out) const {
    driver.capture(in[0], in[1]);
    std::vector<double> y_cur = driver.compute_the_argmax();

    active_t x1_active, x2_active, y1_active, y2_active, res;
    x1_active = in[0];
    x2_active = in[1];
    y1_active = y_cur[0];
    y2_active = y_cur[1];

    ad::derivative(x1_active) = 1.0;
    ad::derivative(x2_active) = 0.0;
    res    = saddle::objective(x1_active, x2_active, y1_active, y2_active);
    out[0] = ad::derivative(res);

    ad::derivative(x1_active) = 0.0;
    ad::derivative(x2_active) = 1.0;
    res    = saddle::objective(x1_active, x2_active, y1_active, y2_active);
    out[1] = ad::derivative(res);
  }

  std::vector<double> compute_the_argmin() const {
    return multivariate_argmin(*this, _start.data());
  }
};

template <template <typename> typename INNER_DRIVER_T>
struct argmin_x_adjoint_driver {
  using adjoint  = ad::adjoint<double>;
  using active_t = ad::adjoint_t<double>;
  INNER_DRIVER_T<double> driver;

  ad::shared_global_tape_ptr<adjoint> _tape;
  std::array<double, 2>               _start;

  argmin_x_adjoint_driver(double const* start) : driver(start), _start() {
    _start[0] = start[0];
    _start[1] = start[1];
  }

  constexpr size_t input_size() const { return 2; }

  void objective(double const* in, double* out) const {
    driver.capture(in[0], in[1]);
    std::vector<double> y_cur = driver.compute_the_argmax();
    *out = saddle::objective(in[0], in[1], y_cur[0], y_cur[1]);
  }

  void gradient(double const* in, double* out) const {
    driver.capture(in[0], in[1]);
    std::vector<double> y_cur = driver.compute_the_argmax();

    active_t x1_active, x2_active, y1_active, y2_active, res;
    x1_active = in[0];
    x2_active = in[1];
    y1_active = y_cur[0];
    y2_active = y_cur[1];

    _tape->reset();
    _tape->register_variable(x1_active);
    _tape->register_variable(x2_active);

    res = saddle::objective(x1_active, x2_active, y1_active, y2_active);
    ad::derivative(res) = 1.0;
    _tape->interpret_adjoint();
    out[0] = ad::derivative(x1_active);
    out[1] = ad::derivative(x2_active);
  }

  std::vector<double> compute_the_argmin() const {
    return multivariate_argmin(*this, _start.data());
  }
};

class SaddleFF : public Function<saddle::Input, saddle::Output> {

  argmin_x_tangent_driver<argmax_y_tangent_driver> argmin_driver;
  argmax_y_tangent_driver<double>                  argmax_driver;
  // input is double[2]
  // output is double[4]
public:
  SaddleFF(saddle::Input input)
      : Function(input), argmin_driver(_input.start),
        argmax_driver(_input.start) {}

  void compute(Output& out) {
    out.resize(4);
    // compute argmin_x max_y f(x, y)
    std::vector<double> x = argmin_driver.compute_the_argmin();
    // write result into output[0], output[1]
    out[0] = x[0];
    out[1] = x[1];
    // compute argmax_y f(x*, y)
    std::vector<double> y = argmax_driver.compute_the_argmax();
    // write result into output[0], output[1]
    out[2] = y[0];
    out[3] = y[1];
  }
};

class SaddleFR : public Function<saddle::Input, saddle::Output> {

  argmin_x_tangent_driver<argmax_y_adjoint_driver> argmin_driver;
  argmax_y_adjoint_driver<double>                  argmax_driver;
  // input is double[2]
  // output is double[4]
public:
  SaddleFR(saddle::Input input)
      : Function(input), argmin_driver(_input.start),
        argmax_driver(_input.start) {}

  void compute(Output& out) {
    out.resize(4);
    // compute argmin_x max_y f(x, y)
    std::vector<double> x = argmin_driver.compute_the_argmin();
    // write result into output[0], output[1]
    out[0] = x[0];
    out[1] = x[1];
    // compute argmax_y f(x*, y)
    std::vector<double> y = argmax_driver.compute_the_argmax();
    // write result into output[0], output[1]
    out[2] = y[0];
    out[3] = y[1];
  }
};

class SaddleRF : public Function<saddle::Input, saddle::Output> {

  argmin_x_adjoint_driver<argmax_y_tangent_driver> argmin_driver;
  argmax_y_tangent_driver<double>                  argmax_driver;
  // input is double[2]
  // output is double[4]
public:
  SaddleRF(saddle::Input input)
      : Function(input), argmin_driver(_input.start),
        argmax_driver(_input.start) {}

  void compute(Output& out) {
    out.resize(4);
    // compute argmin_x max_y f(x, y)
    std::vector<double> x = argmin_driver.compute_the_argmin();
    // write result into output[0], output[1]
    out[0] = x[0];
    out[1] = x[1];
    // compute argmax_y f(x*, y)
    std::vector<double> y = argmax_driver.compute_the_argmax();
    // write result into output[0], output[1]
    out[2] = y[0];
    out[3] = y[1];
  }
};

class SaddleRR : public Function<saddle::Input, saddle::Output> {

  argmin_x_adjoint_driver<argmax_y_adjoint_driver> argmin_driver;
  argmax_y_adjoint_driver<double>                  argmax_driver;
  // input is double[2]
  // output is double[4]
public:
  SaddleRR(saddle::Input input)
      : Function(input), argmin_driver(_input.start),
        argmax_driver(_input.start) {}

  void compute(Output& out) {
    out.resize(4);
    // compute argmin_x max_y f(x, y)
    std::vector<double> x = argmin_driver.compute_the_argmin();
    // write result into output[0], output[1]
    out[0] = x[0];
    out[1] = x[1];
    // compute argmax_y f(x*, y)
    std::vector<double> y = argmax_driver.compute_the_argmax();
    // write result into output[0], output[1]
    out[2] = y[0];
    out[3] = y[1];
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv,
                      {{"rr", function_main<SaddleRR>},
                       {"rf", function_main<SaddleRF>},
                       {"ff", function_main<SaddleFF>},
                       {"fr", function_main<SaddleFR>}});
}
