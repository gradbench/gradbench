module Hello

function square(x)
    return x * x
end

precompile(square, (Float64,))

import ..GradBench
using ..ADTypes: AbstractADType
import ..DifferentiationInterface as DI

abstract type AbstractHello <: GradBench.Experiment end

function GradBench.preprocess(::AbstractHello, input)
    return (; input)
end

struct PrimalHello <: AbstractHello end
(::PrimalHello)(x) = square(x)

struct DIGradientHello{B <: AbstractADType} <: GradBench.Experiment
    backend::B
end

function GradBench.preprocess(g::DIGradientHello, message)
    (; backend) = g
    (; input) = GradBench.preprocess(PrimalHello(), message)
    prep = DI.prepare_derivative(square, backend, zero(input))
    return (; prep, input)
end

function (g::DIGradientHello)(prep, input)
    (; backend) = g
    return DI.derivative(square, prep, backend, input)
end

end
