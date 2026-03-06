module LSE

using ADTypes: AutoMooncake
import Mooncake
import GradBench

backend = AutoMooncake(; config = nothing)

GradBench.register!(
    "lse", Dict(
        "primal" => GradBench.LSE.PrimalLSE(),
        "gradient" => GradBench.LSE.DIGradientLSE(backend)
    )
)

end # module
