module LSE

struct Input
    x::Vector{Float64}
end

function logsumexp(x::Vector{T}) where {T}
    xmax = maximum(x)
    return xmax + log(sum(exp.(x .- xmax)))
end

import ..GradBench

abstract type AbstractLSE <: GradBench.Experiment end

function GradBench.preprocess(::AbstractLSE, message)
    return (Input(convert(Vector{Float64}, message["x"])),)
end

struct PrimalLSE <: AbstractLSE end
(::PrimalLSE)(input) = logsumexp(input.x)


end # module lse
