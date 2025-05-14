module LSE

using Enzyme
import GradBench

struct GradientLSE <: GradBench.LSE.AbstractLSE end

function (::GradientLSE)(input)
    dx = Enzyme.make_zero(input.x)

    if GradBench.OPTIONS["multithreaded"]
        Enzyme.autodiff(
            Reverse, GradBench.LSE.Impure.logsumexp, Active,
            Duplicated(input.x, dx)
        )
    else
        Enzyme.autodiff(
            Reverse, GradBench.LSE.Pure.logsumexp, Active,
            Duplicated(input.x, dx)
        )
    end
    return dx
end

struct PrimalLSE <: GradBench.LSE.AbstractLSE end

function (::PrimalLSE)(input)
    if GradBench.OPTIONS["multithreaded"]
        return GradBench.LSE.Impure.logsumexp(input.x)
    else
        return GradBench.LSE.Pure.logsumexp(input.x)
    end
end

GradBench.register!(
    "lse", Dict(
        "primal" => PrimalLSE(),
        "gradient" => GradientLSE()
    )
)

end # module
