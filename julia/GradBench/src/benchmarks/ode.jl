# Based on
#
# https://github.com/bradbell/cmpad/blob/e375e0606f9b6f6769ea4ce0a57a00463a090539/cpp/include/cmpad/algo/runge_kutta.hpp
#
# originally by Bradley M. Bell <bradbell@seanet.com>, and used here
# under the terms of the EPL-2.0 or GPL-2.0-or-later.
#
# The implementation in cmpad is factored into a generic Runge-Kutta
# module and an instantiation for the specific function under
# consideration. In this implementation, this is all inlined for
# implementation simplicity. This is follows the style use in "ode.hpp"
#
# This module provides two implementations: one that requires
# mutation, and one that does not.

module ODE

import ..GradBench
using ..ADTypes: AbstractADType
import ..DifferentiationInterface as DI

abstract type AbstractODE <: GradBench.Experiment end

function GradBench.preprocess(::AbstractODE, message)
    x = convert(Vector{Float64}, message["x"])
    s = message["s"]
    (; x, s)
end

# An implementation in a pure and vectorised style.
module Pure

using ..ODE

function ode_fun(x::Vector{T}, y::Vector{T}) where {T}
    return [x[1]; x[2:end] .* y[1:end-1]]
end

function runge_kutta(x::Vector{T}, yf::Vector{T}, tf::Float64, s::Int) where {T}
    h = tf / s

    for _ in 1:s
        k1 = ode_fun(x, yf)
        k2 = ode_fun(x, yf .+ (h / 2) .* k1)
        k3 = ode_fun(x, yf .+ (h / 2) .* k2)
        k4 = ode_fun(x, yf .+ h .* k3)

        increment = (h / 6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
        yf = yf .+ increment
    end

    return yf
end

function primal(x::Vector{T}, s::Int) where {T}
    tf = 2.0
    yi = fill(0.0, length(x))
    return runge_kutta(x, yi, tf, s)
end

import ..ODE

struct PrimalODE <: ODE.AbstractODE end
function (::PrimalODE)(x, s)
    return primal(x, s)
end

end # module Pure

# An implementation that uses side effects.
module Impure

using ..ODE

function ode_fun(n, x, y, z)
    z[1] = x[1]
    for i in 2:n
        z[i] = x[i] * y[i-1]
    end
end

function primal(n, xi::AbstractVector{T}, s, yf::Vector{T}) where {T}
    tf = T(2)
    h = tf / T(s)

    k1 = Vector{T}(undef, n)
    k2 = Vector{T}(undef, n)
    k3 = Vector{T}(undef, n)
    k4 = Vector{T}(undef, n)
    y_tmp = Vector{T}(undef, n)

    yf .= T(0)

    for _ in 1:s
        ode_fun(n, xi, yf, k1)

        for i in 1:n
            y_tmp[i] = yf[i] + h * k1[i] / T(2)
        end
        ode_fun(n, xi, y_tmp, k2)

        for i in 1:n
            y_tmp[i] = yf[i] + h * k2[i] / T(2)
        end
        ode_fun(n, xi, y_tmp, k3)

        for i in 1:n
            y_tmp[i] = yf[i] + h * k3[i]
        end
        ode_fun(n, xi, y_tmp, k4)

        for i in 1:n
            yf[i] += h * (k1[i] + T(2) * k2[i] + T(2) * k3[i] + k4[i]) / T(6)
        end
    end
end

import ...GradBench
import ..ODE

struct PrimalODE <: ODE.AbstractODE end
function (::PrimalODE)(x, s)
    output = similar(x)
    n = length(x)

    primal(n, x, s, output)
    return output
end


end # module Impure

struct DIGradientODE{P,B<:AbstractADType} <: GradBench.Experiment
    primal::P
    backend::B
end

function GradBench.preprocess(g::DIGradientODE, message)
    (; primal, backend) = g
    (; x, s) = GradBench.preprocess(primal, message)
    # replace pullback of [0, ..., 0, 1] with gradient of the last component, more optimized for simple backends like ForwardDiff & ReverseDiff
    prep = DI.prepare_gradient(last ∘ primal, backend, zero(x), DI.Constant(s))
    return (; prep, x, s)
end

function (g::DIGradientODE)(prep, x, s)
    (; primal, backend) = g
    return DI.gradient(last ∘ primal, prep, backend, x, DI.Constant(s))
end

end # module ode
