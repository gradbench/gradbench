module LLSQ

using Enzyme
import GradBench

struct GradientLLSQ <: GradBench.LLSQ.AbstractLLSQ end
function (::GradientLLSQ)(input)
    x = input.x
    n = input.n

    dx = Enzyme.make_zero(x)

    Enzyme.autodiff(
        Reverse, GradBench.LLSQ.primal, Active,
        Duplicated(x, dx), Const(n)
    )
    return dx
end

GradBench.register!("llsq", Dict(
    "primal" => GradBench.LLSQ.PrimalLLSQ(),
    "gradient" => GradientLLSQ()
))

end
