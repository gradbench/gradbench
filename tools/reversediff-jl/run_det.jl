module Det

using ADTypes: AutoReverseDiff
import ReverseDiff
import GradBench

primal = GradBench.Det.Impure.PrimalDet()
backend = AutoReverseDiff(; compile = true)

GradBench.register!(
    "det", Dict(
        "primal" => primal,
        "gradient" => GradBench.Det.DIGradientDet(primal, backend)
    )
)

end # module
