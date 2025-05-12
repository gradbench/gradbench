module Hello

function square(x)
    return x * x
end

precompile(square, (Float64,))

import ..GradBench

abstract type AbstractHello <: GradBench.Experiment end

function GradBench.preprocess(::AbstractHello, input)
    (; input)
end

struct PrimalHello <: AbstractHello end
(::PrimalHello)(x) = square(x)

end
