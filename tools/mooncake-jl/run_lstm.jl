module LSTM

using ADTypes: AutoMooncake
import Mooncake
import GradBench

backend = AutoMooncake(; config = nothing)

GradBench.register!(
    "lstm", Dict(
        "objective" => GradBench.LSTM.ObjectiveLSTM(),
        "jacobian" => GradBench.LSTM.DIGradientLSTM(backend)
    )
)

end
