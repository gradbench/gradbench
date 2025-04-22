module LSE

using Enzyme
import GradBench

function gradient(message)
    x = convert(Vector{Float64}, message["x"])

    dx = Enzyme.make_zero(x)

    Enzyme.autodiff(
        Reverse, GradBench.LSE.logsumexp, Active,
        Duplicated(x, dx)
    )
    return dx
end

GradBench.register!(
    "lse", Dict(
        "primal" => GradBench.LSE.primal,
        "gradient" => gradient
    )
)

end # module
