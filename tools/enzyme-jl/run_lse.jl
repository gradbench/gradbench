module ODE

using Enzyme
import GradBench

struct GradientLSE <: GradBench.LSE.AbstractLSE end

function (::GradientLSE)(x)
    dx = Enzyme.make_zero(x)

    Enzyme.autodiff(
        Reverse, GradBench.LSE.logsumexp, Active,
        Duplicated(x, dx)
    )
    return dx
end

GradBench.register!(
    "lse", Dict(
        "primal" => GradBench.LSE.PrimalLSE(),
        "gradient" => GradientLSE()
    )
)

end # module
