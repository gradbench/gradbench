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
    x = convert(Vector{Float64}, message["x"])
    (; x)
end

struct PrimalLSE <: AbstractLSE end
(::PrimalLSE)(x) = logsumexp(x)


end # module lse
