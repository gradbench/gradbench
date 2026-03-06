module GMM

using ADTypes: AutoMooncake
import Mooncake
import GradBench

backend = AutoMooncake(; config = nothing)

GradBench.register!(
    "gmm", Dict(
        "objective" => GradBench.GMM.ObjectiveGMM(),
        "jacobian" => GradBench.GMM.DIGradientGMM(backend)
    )
)

end # module
