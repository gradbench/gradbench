module LSE

import Zygote
import GradBench

const logsumexp = GradBench.LSE.Serial.logsumexp
const PrimalLSE = GradBench.LSE.Serial.PrimalLSE

struct GradientLSE <: GradBench.LSE.AbstractLSE end
function (::GradientLSE)(input)
    z, = Zygote.gradient(logsumexp, input.x)
    return z
end

GradBench.register!("lse", Dict(
    "primal" => PrimalLSE(),
    "gradient" => GradientLSE()
))


end # module
