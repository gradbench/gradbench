#pragma once

#include <codi.hpp>

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

template<int T_dim>
struct CoDiForwardRunnerVec {
  static int constexpr dim = T_dim;

  using Real = codi::RealForwardVec<dim>;
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
      return (size + (size - 1)) / dim; // Round up
    }
  }
};

template<typename T_Real>
struct CoDiReverseRunnerBase {
  using Real = T_Real;

  using Tape = typename Real::Tape;

  Tape& tape = Real::getTape();

  CODI_INLINE void codiStartRecording() {
    tape.setActive();
  }

  CODI_INLINE void codiStopRecording() {
    tape.setPassive();
  }

  CODI_INLINE void codiEval() {
    tape.evaluate();
  }

  CODI_INLINE void codiCleanup() {
    tape.reset(false);
  }

  CODI_INLINE void codiAddInput(Real& value) {
    tape.registerInput(value);
  }

  CODI_INLINE void codiAddOutput(Real& value) {
    tape.registerOutput(value);
  }
};

struct CoDiReverseRunner : public CoDiReverseRunnerBase<codi::RealReverse> {
  using Base = CoDiReverseRunnerBase<codi::RealReverse>;
  using Real = typename Base::Real;

  using Tape = typename CoDiReverseRunnerBase::Tape;
  using Gradient = typename Real::Gradient;

  CODI_INLINE void codiSetGradient(Real& value, Gradient const& grad) {
    value.setGradient(grad);
  }

  CODI_INLINE Gradient codiGetGradient(Real& value) {
    Gradient r = value.getGradient();
    value.gradient() = Gradient();

    return r;
  }
};


struct CoDiReverseRunner2nd : public CoDiReverseRunnerBase<codi::RealReverseGen<codi::RealForward>> {
  using Base = CoDiReverseRunnerBase<codi::RealReverseGen<codi::RealForward>>;
  using Real = typename Base::Real;

  using DH = codi::DerivativeAccess<Real>;

  using Tape = typename CoDiReverseRunnerBase::Tape;
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
    Gradient r = grad;
    grad = Gradient();

    return r;
  }
};

template<int dim>
struct CoDiReverseRunnerVec : public CoDiReverseRunnerBase<codi::RealReverse> {

  using Base = CoDiReverseRunnerBase<codi::RealReverse>;
  using Real = typename Base::Real;

  using Tape = typename Base::Tape;
  using Gradient = typename Real::Real;
  using GradientVec = codi::Direction<Gradient, dim>;

  codi::CustomAdjointVectorHelper<Real, GradientVec > vh;

  CODI_INLINE void codiEval() {
    vh.evaluate();
  }

  CODI_INLINE void codiSetGradient(Real& value, int d, Gradient const& grad) {
    vh.gradient(value.getIdentifier())[d] = grad;
  }

  CODI_INLINE Gradient codiGetGradient(Real& value, int d) {
    GradientVec& vec = vh.gradient(value.getIdentifier());
    Gradient r = vec[d];
    vec[d] = Gradient();

    return r;
  }
};
