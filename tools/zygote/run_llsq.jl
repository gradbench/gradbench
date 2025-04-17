module LLSQ

import Zygote
import GradBench

function gradient(input)
    z, = Zygote.gradient(x -> GradBench.LLSQ.primal(x, input["n"]),
                         convert(Vector{Float64}, input["x"]))
    return z
end

GradBench.register!("llsq", Dict(
    "primal" => input -> GradBench.LLSQ.primal(input["x"], input["n"]),
    "gradient" => gradient
))

end
