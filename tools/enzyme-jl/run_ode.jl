module ODE

using Enzyme
import GradBench

struct GradientODE <: GradBench.ODE.AbstractODE end

import Main: OPTIONS

if OPTIONS["multithreaded"]
    const primal = GradBench.ODE.Parallel.primal
    const PrimalODE = GradBench.ODE.Parallel.PrimalODE
else
    const primal = GradBench.ODE.Serial.primal
    const PrimalODE = GradBench.ODE.Serial.PrimalODE
end

function (::GradientODE)(x, s)
    output = similar(x)
    n = length(x)

    adj = Enzyme.make_zero(output)
    adj[end] = 1

    dx = Enzyme.make_zero(x)

    Enzyme.autodiff(
        Reverse, primal, Const,
        Const(n),
        Duplicated(x, dx),
        Const(s),
        Duplicated(output, adj)
    )
    return dx
end

GradBench.register!(
    "ode", Dict(
        "primal" => PrimalODE(),
        "gradient" => GradientODE()
    )
)

end # module
