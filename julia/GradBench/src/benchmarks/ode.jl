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

using ADTypes: AbstractADType
import DifferentiationInterface as DI

function ode_fun(x, y, z)
    n = length(x)
    z[1] = x[1]
    for i in 2:n
        z[i] = x[i] * y[i-1]
    end
end

function primal!(y::AbstractVector{T}, x::AbstractVector{T}, s::Real) where {T}
    tf = T(2)
    h = tf / T(s)

    k1 = similar(y)
    k2 = similar(y)
    k3 = similar(y)
    k4 = similar(y)
    y_tmp = similar(y)

    y .= T(0)

    for _ in 1:s
        ode_fun(x, y, k1)

        for i in eachindex(y_tmp, y, k1)
            y_tmp[i] = y[i] + h * k1[i] / T(2)
        end
        ode_fun(x, y_tmp, k2)

        for i in eachindex(y_tmp, y, k2)
            y_tmp[i] = y[i] + h * k2[i] / T(2)
        end
        ode_fun(x, y_tmp, k3)

        for i in eachindex(y_tmp, y, k3)
            y_tmp[i] = y[i] + h * k3[i]
        end
        ode_fun(x, y_tmp, k4)

        for i in eachindex(y, k1, k2, k3, k4)
            y[i] += h * (k1[i] + T(2) * k2[i] + T(2) * k3[i] + k4[i]) / T(6)
        end
    end
    return nothing
end

function primal(x::AbstractVector, s::Real)
    y = similar(x)
    primal!(y, x, s)
    return y
end

function gradientlast(backend::AbstractADType, x::AbstractVector, s::Real)
    return DI.gradient(last ∘ primal, backend, x, DI.Constant(s))
end

function parse_input(message::Dict)
    x = convert(Vector{Float64}, message["x"])
    s = message["s"]
    return x, s
end

function primal_from_message(message::Dict)
    x, s = parse_input(message)
    return primal(x, s)
end

function gradientlast_from_message(backend::AbstractADType, message::Dict)
    x, s = parse_input(message)
    return gradientlast(backend, x, s)
end

end # module ode
