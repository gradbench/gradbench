module LSE

using Enzyme
import GradBench

struct GradientLSE <: GradBench.LSE.AbstractLSE end

if OPTIONS["multithreaded"]
    const logsumexp = GradBench.LSE.Parallel.logsumexp
    const PrimalLSE = GradBench.LSE.Parallel.PrimalLSE
else
    const logsumexp = GradBench.LSE.Serial.logsumexp
    const PrimalLSE = GradBench.LSE.Serial.PrimalLSE
end

function (::GradientLSE)(input)
    dx = Enzyme.make_zero(input.x)

    Enzyme.autodiff(
        Reverse, logsumexp, Active,
        Duplicated(input.x, dx)
    )
    return dx
end

GradBench.register!(
    "lse", Dict(
        "primal" => PrimalLSE(),
        "gradient" => GradientLSE()
    )
)

end # module
