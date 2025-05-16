module ODE

using ADTypes: AutoForwardDiff
import ForwardDiff
import GradBench

primal = GradBench.Det.Impure.PrimalDet()
backend = AutoForwardDiff()

GradBench.register!(
    "det", Dict(
        "primal" => primal,
        "gradient" => GradBench.Det.DIGradientDet(primal, backend)
    )
)

end # module
