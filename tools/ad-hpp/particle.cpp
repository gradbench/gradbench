#include <cmath>
#include <tuple>
#include <vector>

#include "ad.hpp"
#include "gradbench/evals/particle.hpp"
#include "gradbench/gd.hpp"
#include "gradbench/main.hpp"

constexpr double PARTICLE_X1_0 = 0.0;
constexpr double PARTICLE_X2_0 = 8.0;
constexpr double PARTICLE_V1_0 = 0.75;
constexpr double PARTICLE_V2_0 = 0.0;
constexpr double DELTA_T       = 1.0e-1;

/**
 * @brief Distance between points (a1, a2) and (b1, b2)
 */
template <typename T>
T dist(T a1, T a2, T b1, T b2) {
  return sqrt((b1 - a1) * (b1 - a1) + (b2 - a2) * (b2 - a2));
}

/**
 * @brief Electric potential defined the the "particle" benchmark.
 * @details This does force a templated expression to be realized here... but it
 * does "work" as well as one might hope
 */
template <typename T>
T potential(T x1, T x2, T w) {
  T distL = dist(x1, x2, (T)10.0, (T)0.0);
  T distR = dist(x1, x2, (T)10.0, (T)(10.0 - w));  // force 10-w to T
  return 1 / distL + 1 / distR;
}

/**
 * @brief Wrapper struct for a tangent mode driver for @ref potential
 */
template <typename T>
struct acceleration_tangent_driver {
  using active_t = ad::tangent_t<T>;

  std::tuple<T, T> operator()(T x1, T x2, T w) const {
    active_t x1_t, x2_t, w_t;
    ad::value(x1_t) = x1;
    ad::value(x2_t) = x2;
    ad::value(w_t)  = w;

    ad::derivative(x1_t) = 1.0;
    ad::derivative(x2_t) = 0.0;
    T dp_dx1             = ad::derivative(potential(x1_t, x2_t, w_t));

    ad::derivative(x1_t) = 0.0;
    ad::derivative(x2_t) = 1.0;
    T dp_dx2             = ad::derivative(potential(x1_t, x2_t, w_t));

    return std::tuple<T, T>({dp_dx1, dp_dx2});
  }
};

/**
 * @brief Wrapper struct for an adjoint mode driver for @ref potential
 */
template <typename T>
struct acceleration_adjoint_driver {
  using adjoint        = ad::adjoint<T>;
  using active_t       = ad::adjoint_t<T>;
  using tape_t         = typename adjoint::tape_t;
  using tape_options_t = typename adjoint::tape_options_t;

  tape_t* _tape;
  acceleration_adjoint_driver() {
    // need to see how good / bad this is
    tape_options_t opts(AD_DEFAULT_TAPE_SIZE);
    _tape = tape_t::create(opts);
  }
  ~acceleration_adjoint_driver() { tape_t::remove(_tape); }
  std::tuple<T, T> operator()(T x1, T x2, T w) const {
    _tape->reset();
    active_t x1_a, x2_a, w_a;
    ad::value(x1_a) = x1;
    ad::value(x2_a) = x2;
    ad::value(w_a)  = w;

    _tape->register_variable(x1_a);
    _tape->register_variable(x2_a);
    _tape->register_variable(w_a);

    active_t p        = potential(x1_a, x2_a, w_a);
    ad::derivative(p) = 1.0;
    _tape->interpret_adjoint();
    return {ad::derivative(x1_a), ad::derivative(x2_a)};
  }
};

template <typename T, typename DRIVER>
std::tuple<T, T, T, T> step_fwd_euler(T x1, T x2, T v1, T v2, T w, double dt,
                                      DRIVER const& driver) {
  T dp_dx1, dp_dx2;
  std::tie(dp_dx1, dp_dx2) = driver(x1, x2, w);
  // compute acceleration as mass * charge * -grad p
  T a1 = -dp_dx1;
  T a2 = -dp_dx2;
  // forward euler step
  x1 += dt * v1;
  x2 += dt * v2;
  v1 += dt * a1;
  v2 += dt * a2;

  return {x1, x2, v1, v2};
}

template <typename T, typename DRIVER>
T compute_particle_intersection_with_xaxis_sqr(T x1, T x2, T v1, T v2,
                                               T const w, double dt,
                                               DRIVER const& driver) {
  T      x1_next, x2_next, v1_next, v2_next;
  size_t counter = 0;
  while (counter < 1'000'000) {
    counter++;
    std::tie(x1_next, x2_next, v1_next, v2_next) =
        step_fwd_euler(x1, x2, v1, v2, w, dt, driver);
    if (x2_next < 0) {
      break;
    }
  }
  T dt_final = -x2 / v2;
  T x1_final = x1 + dt_final * v1;
  return x1_final * x1_final;
}

class ParticleFR : public Function<particle::Input, particle::Output> {
  class optim_wrapper {
    using argmin_active_t = ad::tangent_t<double>;
    /**
     * @brief The driver type for Euler's method when evaluating the objective.
     */
    using euler_obj_driver_t = acceleration_tangent_driver<double>;
    /**
     * @brief The driver type for Euler's method when evaluating the gradient.
     */
    using euler_grad_driver_t = acceleration_adjoint_driver<argmin_active_t>;

    euler_obj_driver_t  accel_driver_obj;
    euler_grad_driver_t accel_driver_grad;
    template <typename T = double, typename DRIVER>
    void _objective(T const* w, T* out, DRIVER const& driver) const {
      T x1_0 = PARTICLE_X1_0;
      T x2_0 = PARTICLE_X2_0;
      T v1_0 = PARTICLE_V1_0;
      T v2_0 = PARTICLE_V2_0;

      *out = compute_particle_intersection_with_xaxis_sqr(
          x1_0, x2_0, v1_0, v2_0, w[0], DELTA_T, driver);
    }

  public:
    size_t input_size() const { return 1; }

    void objective(double const* w, double* out) const {
      _objective(w, out, accel_driver_obj);
    }

    void gradient(double const* w, double* out) const {
      argmin_active_t o_active;
      argmin_active_t w_active = w[0];
      _objective(&w_active, &o_active, accel_driver_grad);
      *out = ad::derivative(o_active);
    }
  };

public:
  ParticleFR(particle::Input& input) : Function(input) {};

  void compute(particle::Output& output) {
    output = multivariate_argmin(optim_wrapper(), &_input.w0)[0];
  }
};

class ParticleRR : public Function<particle::Input, particle::Output> {
  class optim_wrapper {
    using argmin_active_t       = ad::adjoint_t<double>;
    using argmin_adjoint        = ad::adjoint<double>;
    using argmin_tape_t         = argmin_adjoint::tape_t;
    using argmin_tape_options_t = argmin_adjoint::tape_options_t;
    /**
     * @brief The driver type for Euler's method when evaluating the objective.
     */
    using euler_obj_driver_t = acceleration_adjoint_driver<double>;
    /**
     * @brief The driver type for Euler's method when evaluating the gradient.
     */
    using euler_grad_driver_t = acceleration_adjoint_driver<argmin_active_t>;

    euler_obj_driver_t  accel_driver_obj;
    euler_grad_driver_t accel_driver_grad;

    argmin_tape_t* _tape;

    template <typename T = double, typename DRIVER>
    void _objective(T const* w, T* out, DRIVER const& driver) const {
      T x1_0 = PARTICLE_X1_0;
      T x2_0 = PARTICLE_X2_0;
      T v1_0 = PARTICLE_V1_0;
      T v2_0 = PARTICLE_V2_0;

      *out = compute_particle_intersection_with_xaxis_sqr(
          x1_0, x2_0, v1_0, v2_0, w[0], DELTA_T, driver);
    }

  public:
    optim_wrapper() {
      argmin_tape_options_t opts(AD_DEFAULT_TAPE_SIZE);
      _tape = argmin_tape_t::create(opts);
    }
    ~optim_wrapper() { argmin_tape_t::remove(_tape); }

    size_t input_size() const { return 1; }

    void objective(double const* w, double* out) const {
      _objective(w, out, accel_driver_obj);
    }

    void gradient(double const* w, double* out) const {
      argmin_active_t o_active;
      argmin_active_t w_active = w[0];
      _tape->reset();
      _tape->register_variable(w_active);
      _objective(&w_active, &o_active, accel_driver_grad);
      ad::derivative(o_active) = 1.0;
      _tape->interpret_adjoint();
      *out = ad::derivative(w_active);
    }
  };

public:
  ParticleRR(particle::Input& input) : Function(input) {};

  void compute(particle::Output& output) {
    output = multivariate_argmin(optim_wrapper(), &_input.w0)[0];
  }
};

class ParticleFF : public Function<particle::Input, particle::Output> {
  class optim_wrapper {
    using argmin_active_t = ad::tangent_t<double>;
    /**
     * @brief The driver type for Euler's method when evaluating the objective.
     */
    using euler_obj_driver_t = acceleration_tangent_driver<double>;
    /**
     * @brief The driver type for Euler's method when evaluating the gradient.
     */
    using euler_grad_driver_t = acceleration_adjoint_driver<argmin_active_t>;

    euler_obj_driver_t  accel_driver_obj;
    euler_grad_driver_t accel_driver_grad;

    template <typename T = double, typename DRIVER>
    void _objective(T const* w, T* out, DRIVER const& driver) const {
      T x1_0 = PARTICLE_X1_0;
      T x2_0 = PARTICLE_X2_0;
      T v1_0 = PARTICLE_V1_0;
      T v2_0 = PARTICLE_V2_0;

      *out = compute_particle_intersection_with_xaxis_sqr(
          x1_0, x2_0, v1_0, v2_0, w[0], DELTA_T, driver);
    }

  public:
    size_t input_size() const { return 1; }

    void objective(double const* w, double* out) const {
      _objective(w, out, accel_driver_obj);
    }

    void gradient(double const* w, double* out) const {
      argmin_active_t o_active;
      argmin_active_t w_active = w[0];
      _objective(&w_active, &o_active, accel_driver_grad);
      *out = ad::derivative(o_active);
    }
  };

public:
  ParticleFF(particle::Input& input) : Function(input) {};

  void compute(particle::Output& output) {
    output = multivariate_argmin(optim_wrapper(), &_input.w0)[0];
  }
};

class ParticleRF : public Function<particle::Input, particle::Output> {
  class optim_wrapper {
    using argmin_active_t       = ad::adjoint_t<double>;
    using argmin_adjoint        = ad::adjoint<double>;
    using argmin_tape_t         = argmin_adjoint::tape_t;
    using argmin_tape_options_t = argmin_adjoint::tape_options_t;
    /**
     * @brief The driver type for Euler's method when evaluating the objective.
     */
    using euler_obj_driver_t = acceleration_tangent_driver<double>;
    /**
     * @brief The driver type for Euler's method when evaluating the gradient.
     */
    using euler_grad_driver_t = acceleration_adjoint_driver<argmin_active_t>;

    euler_obj_driver_t  accel_driver_obj;
    euler_grad_driver_t accel_driver_grad;
    argmin_tape_t*      _tape;

    template <typename T = double, typename DRIVER>
    void _objective(T const* w, T* out, DRIVER const& driver) const {
      T x1_0 = PARTICLE_X1_0;
      T x2_0 = PARTICLE_X2_0;
      T v1_0 = PARTICLE_V1_0;
      T v2_0 = PARTICLE_V2_0;

      *out = compute_particle_intersection_with_xaxis_sqr(
          x1_0, x2_0, v1_0, v2_0, w[0], DELTA_T, driver);
    }

  public:
    optim_wrapper() {
      argmin_tape_options_t opts(AD_DEFAULT_TAPE_SIZE);
      _tape = argmin_tape_t::create(opts);
    }
    ~optim_wrapper() { argmin_tape_t::remove(_tape); }

    size_t input_size() const { return 1; }

    void objective(double const* w, double* out) const {
      _objective(w, out, accel_driver_obj);
    }

    void gradient(double const* w, double* out) const {
      argmin_active_t o_active;
      argmin_active_t w_active = w[0];
      _tape->reset();
      _tape->register_variable(w_active);
      _objective(&w_active, &o_active, accel_driver_grad);
      ad::derivative(o_active) = 1.0;
      _tape->interpret_adjoint();
      *out = ad::derivative(w_active);
    }
  };

public:
  ParticleRF(particle::Input& input) : Function(input) {};

  void compute(particle::Output& output) {
    output = multivariate_argmin(optim_wrapper(), &_input.w0)[0];
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv,
                      {{"rr", function_main<ParticleRR>},
                       {"rf", function_main<ParticleRF>},
                       {"ff", function_main<ParticleFF>},
                       {"fr", function_main<ParticleFR>}});
}
