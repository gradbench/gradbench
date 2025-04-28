module ODE

import Zygote
import GradBench

struct GradientODE <: GradBench.ODE.AbstractODE end
function (::GradientODE)(x, s)

    z, = Zygote.gradient(x -> GradBench.ODE.Pure.primal(x, s)[end],
                         x)
    return z
end

GradBench.register!(
    "ode", Dict(
        "primal" => GradBench.ODE.Pure.PrimalODE(),
        "gradient" => GradientODE()
    )
)

end # module
