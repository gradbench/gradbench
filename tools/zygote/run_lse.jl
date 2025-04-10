module LSE

import Zygote
import GradBench

function gradient(input)
    x = convert(Vector{Float64}, input["x"])
    z, = Zygote.gradient(GradBench.LSE.logsumexp, x)
    return z
end

GradBench.register!("lse", Dict(
    "primal" => GradBench.LSE.primal,
    "gradient" => gradient
))


end # module
