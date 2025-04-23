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
struct potential_tangent_driver {
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
struct potential_adjoint_driver {
  using adjoint        = ad::adjoint<T>;
  using active_t       = ad::adjoint_t<T>;
  using tape_t         = typename adjoint::tape_t;
  using tape_options_t = typename adjoint::tape_options_t;

  static int _existing_drivers;

  potential_adjoint_driver() {
    _existing_drivers++;
    tape_options_t opts(1024 * 8);
    if (adjoint::global_tape == nullptr) {
      adjoint::global_tape = tape_t::create(opts);
    }
  }

  ~potential_adjoint_driver() {
    _existing_drivers--;
    if (_existing_drivers == 0) {
      tape_t::remove(adjoint::global_tape);
    }
  }

  std::tuple<T, T> operator()(T x1, T x2, T w) const {
    tape_t* gtape = adjoint::global_tape;
    gtape->reset();
    active_t x1_a, x2_a, w_a;
    ad::value(x1_a) = x1;
    ad::value(x2_a) = x2;
    ad::value(w_a)  = w;

    gtape->register_variable(x1_a);
    gtape->register_variable(x2_a);
    gtape->register_variable(w_a);

    active_t p        = potential(x1_a, x2_a, w_a);
    ad::derivative(p) = 1.0;
    gtape->interpret_adjoint();
    return {ad::derivative(x1_a), ad::derivative(x2_a)};
  }
};

template <typename T>
int potential_adjoint_driver<T>::_existing_drivers = 0;

template <typename T, typename POTENTIAL_DRIVER>
std::tuple<T, T, T, T>
step_fwd_euler(T x1, T x2, T v1, T v2, T w, double dt,
               POTENTIAL_DRIVER const& grad_x_potential) {
  T dp_dx1, dp_dx2, a1, a2;
  std::tie(dp_dx1, dp_dx2) = grad_x_potential(x1, x2, w);
  // compute acceleration as mass * charge * -grad p
  a1 = -dp_dx1;
  a2 = -dp_dx2;
  // euler step x^{k+1} = x^k+dt*v^k; v^{k+1} = v^k+dt*a^k
  return {x1 + dt * v1, x2 + dt * v2, v1 + dt * a1, v2 + dt * a2};
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
    x1 = x1_next;
    x2 = x2_next;
    v1 = v1_next;
    v2 = v2_next;
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
    using euler_obj_driver_t = potential_adjoint_driver<double>;
    /**
     * @brief The driver type for Euler's method when evaluating the gradient.
     */
    using euler_grad_driver_t = potential_adjoint_driver<argmin_active_t>;

    euler_obj_driver_t  potential_driver_objective;
    euler_grad_driver_t potential_driver_gradient;
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
      _objective(w, out, potential_driver_objective);
    }

    void gradient(double const* w, double* out) const {
      argmin_active_t o_active;
      argmin_active_t w_active = w[0];
      ad::derivative(w_active) = 1.0;
      _objective(&w_active, &o_active, potential_driver_gradient);
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
    using euler_obj_driver_t = potential_adjoint_driver<double>;
    /**
     * @brief The driver type for Euler's method when evaluating the gradient.
     */
    using euler_grad_driver_t = potential_adjoint_driver<argmin_active_t>;

    euler_obj_driver_t  potential_driver_objective;
    euler_grad_driver_t potential_driver_gradient;

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
      argmin_adjoint::global_tape = argmin_tape_t::create(opts);
    }
    ~optim_wrapper() { argmin_tape_t::remove(argmin_adjoint::global_tape); }

    size_t input_size() const { return 1; }

    void objective(double const* w, double* out) const {
      _objective(w, out, potential_driver_objective);
    }

    void gradient(double const* w, double* out) const {
      argmin_active_t o_active;
      argmin_active_t w_active = w[0];

      argmin_tape_t* gtape = argmin_adjoint::global_tape;
      gtape->reset();
      gtape->register_variable(w_active);

      _objective(&w_active, &o_active, potential_driver_gradient);

      ad::derivative(o_active) = 1.0;
      gtape->interpret_adjoint();
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
    using euler_obj_driver_t = potential_tangent_driver<double>;
    /**
     * @brief The driver type for Euler's method when evaluating the gradient.
     */
    using euler_grad_driver_t = potential_tangent_driver<argmin_active_t>;

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
      ad::derivative(w_active) = 1.0;
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
    using euler_obj_driver_t = potential_tangent_driver<double>;
    /**
     * @brief The driver type for Euler's method when evaluating the gradient.
     */
    using euler_grad_driver_t = potential_tangent_driver<argmin_active_t>;

    euler_obj_driver_t  potential_driver_objective;
    euler_grad_driver_t potential_driver_gradient;

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
      argmin_adjoint::global_tape = argmin_tape_t::create(opts);
    }
    ~optim_wrapper() { argmin_tape_t::remove(argmin_adjoint::global_tape); }

    size_t input_size() const { return 1; }

    void objective(double const* w, double* out) const {
      _objective(w, out, potential_driver_objective);
    }

    void gradient(double const* w, double* out) const {
      argmin_active_t o_active;
      argmin_active_t w_active = w[0];
      argmin_tape_t*  gtape    = argmin_adjoint::global_tape;
      gtape->reset();
      gtape->register_variable(w_active);
      _objective(&w_active, &o_active, potential_driver_gradient);
      ad::derivative(o_active) = 1.0;
      gtape->interpret_adjoint();
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
