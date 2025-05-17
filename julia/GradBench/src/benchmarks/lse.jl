module LSE

import ..GradBench

struct Input
    x::Vector{Float64}
end

abstract type AbstractLSE <: GradBench.Experiment end

function GradBench.preprocess(::AbstractLSE, message)
    (Input(convert(Vector{Float64}, message["x"])),)
end

# A sequential implementation in a vectorised style.
module Serial

using ..LSE

function logsumexp(x::Vector{T}) where {T}
    xmax = maximum(x)
    return xmax + log(sum(exp.(x .- xmax)))
end

struct PrimalLSE <: LSE.AbstractLSE end
(::PrimalLSE)(input) = logsumexp(input.x)

end # Serial

# A multithreaded implementation using a functional parallel style
module Parallel

using ..LSE

using OhMyThreads: treduce, tmapreduce

function logsumexp(x::Vector{T}) where {T}
    xmax = treduce(max, x)
    return xmax + log(tmapreduce(x->exp(x-xmax), +, x))
end

struct PrimalLSE <: LSE.AbstractLSE end
(::PrimalLSE)(input) = logsumexp(input.x)

end # module Parallel

end # module lse
