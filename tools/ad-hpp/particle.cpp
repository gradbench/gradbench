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

// This next line assumes C++17; otherwise, replace it with
// your own string view implementation
#include <string_view>

template <typename T>
constexpr std::string_view type_name();

template <>
constexpr std::string_view type_name<void>() {
  return "void";
}

namespace pretty_typename_detail {

using type_name_prober = void;

template <typename T>
constexpr std::string_view wrapped_type_name() {
#if __cplusplus >= 202002L
  return std::source_location::current().function_name();
#else
#if defined(__clang__) || defined(__GNUC__)
  return __PRETTY_FUNCTION__;
#elif defined(_MSC_VER)
  return __FUNCSIG__;
#else
#error "Unsupported compiler"
#endif
#endif  // __cplusplus >= 202002L
}

constexpr std::size_t wrapped_type_name_prefix_length() {
  return wrapped_type_name<type_name_prober>().find(
      type_name<type_name_prober>());
}

constexpr std::size_t wrapped_type_name_suffix_length() {
  return wrapped_type_name<type_name_prober>().length() -
         wrapped_type_name_prefix_length() -
         type_name<type_name_prober>().length();
}

}  // namespace pretty_typename_detail

template <typename T>
constexpr std::string_view type_name() {
  constexpr auto wrapped_name = pretty_typename_detail::wrapped_type_name<T>();
  constexpr auto prefix_length =
      pretty_typename_detail::wrapped_type_name_prefix_length();
  constexpr auto suffix_length =
      pretty_typename_detail::wrapped_type_name_suffix_length();
  constexpr auto type_name_length =
      wrapped_name.length() - prefix_length - suffix_length;
  return wrapped_name.substr(prefix_length, type_name_length);
}

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
  return 1.0 / distL + 1.0 / distR;
}

/**
 * @brief Wrapper struct for a tangent mode driver for @ref potential
 */
template <typename T>
struct potential_tangent_driver {
  using active_t = ad::tangent_t<T>;

  potential_tangent_driver() {
    std::cerr << "default constructing driver: " << type_name<decltype(*this)>()
              << "\n";
  }

  ~potential_tangent_driver() {
    std::cerr << "destroying driver: " << type_name<decltype(*this)>() << "\n";
  }

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

// This should be added to the next release of ad.hpp
template <typename ADJOINT>
class smart_global_tape_pointer {
  using tape_t = typename ADJOINT::tape_t;
  static unsigned int _refs;

public:
  smart_global_tape_pointer(typename ADJOINT::tape_options_t const& opts) {
    std::cerr << "constructing smart global tape ptr for tape with options: "
              << type_name<decltype(*this)>() << "\n";
    if (_refs == 0) {
      std::cerr << "  initializing tape.\n";
      ADJOINT::global_tape = tape_t::create(opts);
    }
    _refs++;
    std::cerr << "  total number of refs to global tape is " << _refs << "\n";
  }

  smart_global_tape_pointer() {
    std::cerr << "constructing smart global tape ptr for tape with NO options: "
              << type_name<decltype(*this)>() << "\n";
    if (_refs == 0) {
      std::cerr << "  initializing tape.\n";
      ADJOINT::global_tape = tape_t::create();
    }
    _refs++;
    std::cerr << "  total number of refs to global tape is " << _refs << "\n";
  }

  smart_global_tape_pointer(smart_global_tape_pointer const& other) {
    std::cerr << "constructing smart global tape ptr for tape from other: "
              << type_name<decltype(*this)>() << "\n";
    _refs++;
    std::cerr << "  total number of refs to global tape is " << _refs << "\n";
  }

  ~smart_global_tape_pointer() {
    std::cerr << "destructing smart global tape ptr for type: "
              << type_name<decltype(*this)>() << "\n";
    _refs--;
    std::cerr << "  total number of refs to global tape is " << _refs << "\n";
  }

  tape_t* get() { return ADJOINT::global_tape; }
  tape_t& operator*() const { return *ADJOINT::global_tape; }
  tape_t* operator->() const { return ADJOINT::global_tape; }
};

template <typename T>
unsigned int smart_global_tape_pointer<T>::_refs = 0;

/**
 * @brief Wrapper struct for an adjoint mode driver for @ref potential
 */
template <typename T>
struct potential_adjoint_driver {
  using adjoint        = ad::adjoint<T>;
  using active_t       = ad::adjoint_t<T>;
  using tape_t         = typename adjoint::tape_t;
  using tape_options_t = typename adjoint::tape_options_t;

  smart_global_tape_pointer<adjoint> gtape;

  potential_adjoint_driver() : gtape(tape_options_t(AD_DEFAULT_TAPE_SIZE)) {
    std::cerr << "default constructing driver: " << type_name<decltype(*this)>()
              << "\n";
  }

  ~potential_adjoint_driver() {
    std::cerr << "destroying driver: " << type_name<decltype(*this)>() << "\n";
  }

  std::tuple<T, T> operator()(T x1, T x2, T w) const {
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

  optim_wrapper objective_wrapper;

public:
  ParticleFR(particle::Input& input) : Function(input) {};

  void compute(particle::Output& output) {
    output = multivariate_argmin(objective_wrapper, &_input.w0)[0];
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

    smart_global_tape_pointer<argmin_adjoint> argmin_global_tape;

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
    optim_wrapper()
        : argmin_global_tape(argmin_tape_options_t(AD_DEFAULT_TAPE_SIZE)) {}

    size_t input_size() const { return 1; }

    void objective(double const* w, double* out) const {
      _objective(w, out, potential_driver_objective);
    }

    void gradient(double const* w, double* out) const {
      argmin_active_t o_active;
      argmin_active_t w_active = w[0];

      argmin_global_tape->reset();
      argmin_global_tape->register_variable(w_active);

      _objective(&w_active, &o_active, potential_driver_gradient);

      ad::derivative(o_active) = 1.0;
      argmin_global_tape->interpret_adjoint();
      *out = ad::derivative(w_active);
    }
  };

  optim_wrapper objective_wrapper;

public:
  ParticleRR(particle::Input& input) : Function(input) {};

  void compute(particle::Output& output) {
    output = multivariate_argmin(objective_wrapper, &_input.w0)[0];
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

  optim_wrapper objective_wrapper;

public:
  ParticleFF(particle::Input& input) : Function(input) {};

  void compute(particle::Output& output) {
    output = multivariate_argmin(objective_wrapper, &_input.w0)[0];
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

    smart_global_tape_pointer<argmin_adjoint> gtape;

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
    optim_wrapper() : gtape(argmin_tape_options_t(AD_DEFAULT_TAPE_SIZE)) {}

    size_t input_size() const { return 1; }

    void objective(double const* w, double* out) const {
      _objective(w, out, potential_driver_objective);
    }

    void gradient(double const* w, double* out) const {
      argmin_active_t o_active;
      argmin_active_t w_active = w[0];
      gtape->reset();
      gtape->register_variable(w_active);
      _objective(&w_active, &o_active, potential_driver_gradient);
      ad::derivative(o_active) = 1.0;
      gtape->interpret_adjoint();
      *out = ad::derivative(w_active);
    }
  };

  optim_wrapper objective_wrapper;

public:
  ParticleRF(particle::Input& input) : Function(input) {};

  void compute(particle::Output& output) {
    output = multivariate_argmin(objective_wrapper, &_input.w0)[0];
  }
};

int main(int argc, char* argv[]) {
  return generic_main(argc, argv,
                      {{"rr", function_main<ParticleRR>},
                       {"rf", function_main<ParticleRF>},
                       {"ff", function_main<ParticleFF>},
                       {"fr", function_main<ParticleFR>}});
}
