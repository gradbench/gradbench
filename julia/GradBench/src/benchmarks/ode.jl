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

module ODE

struct Input
    x::Vector{Float64}
    s::Int
end

function ode_fun(n, x, y, z)
    z[1] = x[1]
    for i in 2:n
        z[i] = x[i] * y[i-1]
    end
end

function primal(n, xi::Vector{T}, s, yf::Vector{T}) where {T}
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

import ..GradBench

abstract type AbstractODE <: GradBench.Experiment end

function GradBench.preprocess(::AbstractODE, message)
    x = convert(Vector{Float64}, message["x"])
    s = message["s"]
    (; x, s)
end

struct PrimalODE <: AbstractODE end
function (::PrimalODE)(x, s)
    output = similar(x)
    n = length(x)

    primal(n, x, s, output)
    return output
end


end # module ode
