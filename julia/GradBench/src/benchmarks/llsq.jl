module LLSQ

import GradBench

struct Input
    x::Vector{Float64}
    n::Int64
end

abstract type AbstractLLSQ <: GradBench.Experiment end

function GradBench.preprocess(::AbstractLLSQ, message)
    x = convert(Vector{Float64}, message["x"])
    n = message["n"]
    return (Input(x, n),)
end

function t(i::Int64, n::Int64)
    return -1 + (Float64(i) * 2) / (Float64(n) - 1)
end

function primal(x::Vector{T}, n::Int64) where {T}
    m = length(x)

    f(i) = begin
        ti = t(i, n)
        inner_sum = 0.0
        mul = 1.0
        for j in 1:m
            inner_sum += -x[j] * mul
            mul *= ti
        end
        return (sign(ti) + inner_sum)^2
    end

    return sum(f(i) for i in 0:n-1) / 2
end

struct PrimalLLSQ <: AbstractLLSQ end
function (::PrimalLLSQ)(input)
    return primal(input.x, input.n)
end


end
