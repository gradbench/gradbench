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

end # module ode
