module ODE

using ADTypes: AutoReverseDiff
import ReverseDiff
import GradBench

primal = GradBench.ODE.Impure.PrimalODE()
backend = AutoReverseDiff(; compile = false)  # compilation is useless when there are constant arguments, we can't afford to hardcode them in the tape

GradBench.register!(
    "ode", Dict(
        "primal" => primal,
        "gradient" => GradBench.ODE.DIGradientODE(primal, backend)
    )
)

end # module
