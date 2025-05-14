module LSE

import Zygote
import GradBench

struct GradientLSE <: GradBench.LSE.AbstractLSE end
function (::GradientLSE)(input)
    z, = Zygote.gradient(GradBench.LSE.Pure.logsumexp, input.x)
    return z
end

GradBench.register!("lse", Dict(
    "primal" => GradBench.LSE.Pure.PrimalLSE(),
    "gradient" => GradientLSE()
))


end # module
