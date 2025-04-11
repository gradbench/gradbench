module ODE

using ADTypes: AutoMooncake
import Mooncake
import GradBench

GradBench.register!(
    "ode", Dict(
        "primal" => GradBench.ODE.primal_from_message,
        "gradient" => (GradBench.ODE.gradientlast_from_message, AutoMooncake(; config=nothing)),
    )
)

end # module
