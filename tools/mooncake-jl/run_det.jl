module Det

using ADTypes: AutoMooncake
import Mooncake
import GradBench

primal = GradBench.Det.Impure.PrimalDet()
backend = AutoMooncake(; config = nothing)

GradBench.register!(
    "det", Dict(
        "primal" => primal,
        "gradient" => GradBench.Det.DIGradientDet(primal, backend)
    )
)

end # module
