module LLSQ

using Enzyme
import GradBench

function gradient(message)
    x = convert(Vector{Float64}, message["x"])
    n = message["n"]

    dx = Enzyme.make_zero(x)

    Enzyme.autodiff(
        Reverse, GradBench.LLSQ.primal, Active,
        Duplicated(x, dx), Const(n)
    )
    return dx
end

GradBench.register!("llsq", Dict(
    "primal" => input -> GradBench.LLSQ.primal(input["x"], input["n"]),
    "gradient" => gradient
))

end
