module ODE

using ADTypes: AutoForwardDiff
import ForwardDiff
import GradBench

primal = GradBench.ODE.Serial.PrimalODE()
backend = AutoForwardDiff()

GradBench.register!(
    "ode", Dict(
        "primal" => primal,
        "gradient" => GradBench.ODE.DIGradientODE(primal, backend)
    )
)

end # module
