module LSE

using Enzyme
import GradBench

struct GradientLSE <: GradBench.LSE.AbstractLSE end

function (::GradientLSE)(input)
    dx = Enzyme.make_zero(input.x)

    Enzyme.autodiff(
        Reverse, GradBench.LSE.logsumexp, Active,
        Duplicated(input.x, dx)
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
