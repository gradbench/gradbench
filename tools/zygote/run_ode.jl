module ODE

import Zygote
import GradBench

function primal(message)
    # TODO: move the message parsing into the harness
    x = convert(Vector{Float64}, message["x"])
    s = message["s"]

    return GradBench.ODE.Pure.primal(x, s)
end

function gradient(message)
    x = convert(Vector{Float64}, message["x"])
    s = message["s"]

    z, = Zygote.gradient(x -> GradBench.ODE.Pure.primal(x, s)[end],
                         x)
    return z
end

GradBench.register!(
    "ode", Dict(
        "primal" => primal,
        "gradient" => gradient
    )
)

end # module
