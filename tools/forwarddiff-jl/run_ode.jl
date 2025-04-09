module ODE

using ADTypes: AutoForwardDiff
import ForwardDiff
import GradBench

GradBench.register!(
    "ode", Dict(
        "primal" => GradBench.ODE.primal_from_message,
        "gradient" => GradBench.ODE.gradientlast_from_message,
        "backend" => AutoForwardDiff(),
    )
)

end # module
