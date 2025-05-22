module ODE

using ADTypes: AutoMooncake
import Mooncake
import GradBench

primal = GradBench.ODE.Serial.PrimalODE()
backend = AutoMooncake(; config=nothing)

GradBench.register!(
    "ode", Dict(
        "primal" => primal,
        "gradient" => GradBench.ODE.DIGradientODE(primal, backend)
    )
)

end # module
