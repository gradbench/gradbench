module LSE

import ..GradBench
using ..ADTypes: AbstractADType
import ..DifferentiationInterface as DI

struct Input
    x::Vector{Float64}
end

function logsumexp(x::Vector{T}) where {T}
    xmax = maximum(x)
    return xmax + log(sum(exp.(x .- xmax)))
end

abstract type AbstractLSE <: GradBench.Experiment end

function GradBench.preprocess(::AbstractLSE, message)
    return (; input = Input(convert(Vector{Float64}, message["x"])))
end

struct PrimalLSE <: AbstractLSE end
(::PrimalLSE)(input) = logsumexp(input.x)

struct DIGradientLSE{B <: AbstractADType} <: GradBench.Experiment
    backend::B
end

function GradBench.preprocess(g::DIGradientLSE, message)
    (; backend) = g
    (; input) = GradBench.preprocess(PrimalLSE(), message)
    prep = DI.prepare_gradient(logsumexp, backend, input.x)
    return (; prep, input)
end

function (g::DIGradientLSE)(prep, input)
    (; backend) = g
    return DI.gradient(logsumexp, prep, backend, input.x)
end

end # module lse
