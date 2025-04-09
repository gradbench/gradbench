module LogSumExp

import Zygote
import GradBench

function gradient(input)
    x = convert(Vector{Float64}, input["x"])
    z, = Zygote.gradient(GradBench.LogSumExp.logsumexp, x)
    return z
end

GradBench.register!("logsumexp", Dict(
    "primal" => GradBench.LogSumExp.primal,
    "gradient" => gradient
))


end # module
