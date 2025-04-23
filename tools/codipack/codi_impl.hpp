#pragma once

#include <codi.hpp>

namespace type_details {

#ifndef CODI_TYPE
#define CODI_TYPE codi::RealReverse
#endif

#define COMBINE(A, B) COMBINE2(A, B)
#define COMBINE2(A, B) A##B

#define CODI_TYPE_VEC COMBINE(CODI_TYPE, Vec)
#define CODI_TYPE_GEN COMBINE(CODI_TYPE, Gen)

using Base = CODI_TYPE;

template <int dim>
using Vec = CODI_TYPE_VEC<dim>;

template <typename Real, typename Gradient = Real>
using Gen = CODI_TYPE_GEN<Real, Gradient>;

#undef CODI_TYPE
#undef CODI_TYPE_VEC
#undef CODI_TYPE_GEN
#undef COMBINE
#undef COMBINE2
}  // namespace type_details

struct CoDiForwardRunner {
  using Real = codi::RealForward;

  using Gradient = typename Real::Gradient;

  CODI_INLINE void codiSetGradient(Real& value, Gradient const& grad) {
    value.setGradient(grad);
  }

  CODI_INLINE Gradient codiGetGradient(Real& value) {
    return value.getGradient();
  }
};

template <int T_dim>
struct CoDiForwardRunnerVec {
  static int constexpr dim = T_dim;

  using Real     = codi::RealForwardVec<dim>;
  using Gradient = typename Real::Real;

  CODI_INLINE void codiSetGradient(Real& value, int d, Gradient const& grad) {
    value.gradient()[d] = grad;
  }

  CODI_INLINE Gradient codiGetGradient(Real& value, int d) {
    return value.getGradient()[d];
  }

  CODI_INLINE int codiGetRuns(int size) {
    if (0 == size) {
      return 0;
    } else {
      return (size + (size - 1)) / dim;  // Round up
    }
  }
};

template <typename T_Real>
struct CoDiReverseRunnerBase {
  using Real = T_Real;

  using Tape = typename Real::Tape;

  Tape& tape = Real::getTape();

  CODI_INLINE void codiStartRecording() { tape.setActive(); }

  CODI_INLINE void codiStopRecording() { tape.setPassive(); }

  CODI_INLINE void codiEval() { tape.evaluate(); }

  CODI_INLINE void codiCleanup() { tape.reset(false); }

  CODI_INLINE void codiAddInput(Real& value) { tape.registerInput(value); }

  CODI_INLINE void codiAddOutput(Real& value) { tape.registerOutput(value); }
};

struct CoDiReverseRunner : public CoDiReverseRunnerBase<type_details::Base> {
  using Base = CoDiReverseRunnerBase<type_details::Base>;
  using Real = typename Base::Real;

  using Tape     = typename CoDiReverseRunnerBase::Tape;
  using Gradient = typename Real::Gradient;

  CODI_INLINE void codiSetGradient(Real& value, Gradient const& grad) {
    value.setGradient(grad);
  }

  CODI_INLINE Gradient codiGetGradient(Real& value) {
    Gradient r       = value.getGradient();
    value.gradient() = Gradient();

    return r;
  }
};

struct CoDiReverseRunner2nd
    : public CoDiReverseRunnerBase<type_details::Gen<codi::RealForward>> {
  using Base = CoDiReverseRunnerBase<type_details::Gen<codi::RealForward>>;
  using Real = typename Base::Real;

  using DH = codi::DerivativeAccess<Real>;

  using Tape     = typename CoDiReverseRunnerBase::Tape;
  using Gradient = codi::RealTraits::PassiveReal<Real>;

  CODI_INLINE void codiSetGradient(Real& value, Gradient const& grad) {
    DH::setAllDerivativesReverse(value, 1, grad);
  }

  CODI_INLINE void codiAddInput(Real& value) {
    DH::setAllDerivativesForward(value, 1, 1.0);
    Base::codiAddInput(value);
  }

  CODI_INLINE Gradient codiGetGradient(Real& value, size_t order, size_t pos) {
    Gradient& grad = DH::derivative(value, order, pos);
    Gradient  r    = grad;
    grad           = Gradient();

    return r;
  }
};

template <int dim>
struct CoDiReverseRunnerVec : public CoDiReverseRunnerBase<type_details::Base> {

  using Base = CoDiReverseRunnerBase<type_details::Base>;
  using Real = typename Base::Real;

  using Tape        = typename Base::Tape;
  using Gradient    = typename Real::Real;
  using GradientVec = codi::Direction<Gradient, dim>;

  codi::CustomAdjointVectorHelper<Real, GradientVec> vh;

  CODI_INLINE void codiEval() { vh.evaluate(); }

  CODI_INLINE void codiSetGradient(Real& value, int d, Gradient const& grad) {
    vh.gradient(value.getIdentifier())[d] = grad;
  }

  CODI_INLINE Gradient codiGetGradient(Real& value, int d) {
    GradientVec& vec = vh.gradient(value.getIdentifier());
    Gradient     r   = vec[d];
    vec[d]           = Gradient();

    return r;
  }
};
