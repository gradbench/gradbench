module LLSQ

using ADTypes: AutoMooncake
import Mooncake
import GradBench

backend = AutoMooncake(; config = nothing)

GradBench.register!(
    "llsq", Dict(
        "primal" => GradBench.LLSQ.PrimalLLSQ(),
        "gradient" => GradBench.LLSQ.DIGradientLLSQ(backend)
    )
)

end # module
