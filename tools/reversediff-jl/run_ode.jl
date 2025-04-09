module ODE

using ADTypes: AutoReverseDiff
import ReverseDiff
import GradBench

GradBench.register!(
    "ode", Dict(
        "primal" => GradBench.ODE.primal_from_message,
        "gradient" => (GradBench.ODE.gradientlast_from_message, AutoReverseDiff()),
    )
)

end # module
