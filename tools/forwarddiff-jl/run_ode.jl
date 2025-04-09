module ODE

import DifferentiationInterface as DI
import ForwardDiff
import GradBench

function primal(message)
    # TODO: move the message parsing into the harness
    x = convert(Vector{Float64}, message["x"])
    s = message["s"]

    output = similar(x)
    n = length(x)

    GradBench.ODE.primal(n, x, s, output)
    return output
end

function lastprimal(x, s)  # TODO: put in GradBench
    output = similar(x)
    n = length(x)
    GradBench.ODE.primal(n, x, s, output)
    return last(output)
end

function gradient(message)
    x = convert(Vector{Float64}, message["x"])
    s = message["s"]
    grad = DI.gradient(lastprimal, DI.AutoForwardDiff(), x, DI.Constant(s))
    return grad
end

GradBench.register!(
    "ode", Dict(
        "primal" => primal,
        "gradient" => gradient
    )
)

end # module
