module ODE

using Enzyme
import GradBench

function gradient(message)
    x = convert(Vector{Float64}, message["x"])

    dx = Enzyme.make_zero(x)

    Enzyme.autodiff(
        Reverse, GradBench.LogSumExp.logsumexp, Active,
        Duplicated(x, dx)
    )
    return dx
end

GradBench.register!(
    "logsumexp", Dict(
        "primal" => GradBench.LogSumExp.primal,
        "gradient" => gradient
    )
)

end # module
