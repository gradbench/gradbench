module ODE

import Zygote
import GradBench

function primal(message)
    x = convert(Vector{Float64}, message["x"])
    s = message["s"]

    output = similar(x)
    n = length(x)
    return GradBench.ODE.primal(n, x, s, output)
end

function gradient(message)
    x = convert(Vector{Float64}, message["x"])
    s = message["s"]

    _, back = Zygote.pullback(x) do x
        n = length(x)
        output = similar(x)
        GradBench.ODE.primal(n, x, s, output)
        return output
    end

    adj = zero(x)
    adj[end] = 1
    back(adj)
end

GradBench.register!("ode", Dict(
    "primal" => primal,
    "gradient" => gradient
))

end # module
