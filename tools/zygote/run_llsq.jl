module LLSQ

import Zygote
import GradBench

struct GradientLLSQ <: GradBench.LLSQ.AbstractLLSQ end
function (::GradientLLSQ)(input)
    z, = Zygote.gradient(
        x -> GradBench.LLSQ.primal(x, input.n),
        input.x
    )
    return z
end

GradBench.register!(
    "llsq", Dict(
        "primal" => GradBench.LLSQ.PrimalLLSQ(),
        "gradient" => GradientLLSQ()
    )
)

end
