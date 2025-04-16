module ODE

using Enzyme
import GradBench

struct GradientODE <: GradBench.ODE.AbstractODE end
function (::GradientODE)(x, s)
    output = similar(x)
    n = length(x)

    adj = Enzyme.make_zero(output)
    adj[end] = 1

    dx = Enzyme.make_zero(x)

    Enzyme.autodiff(
        Reverse, GradBench.ODE.primal, Const,
        Const(n),
        Duplicated(x, dx),
        Const(s),
        Duplicated(output, adj)
    )
    return dx
end

GradBench.register!(
    "ode", Dict(
        "primal" => GradBench.ODE.PrimalODE(),
        "gradient" => GradientODE()
    )
)

end # module
