module Hello

using ADTypes: AutoForwardDiff
import ForwardDiff
import GradBench

backend = AutoForwardDiff()

GradBench.register!(
    "hello", Dict(
        "square" => GradBench.Hello.PrimalHello(),
        "double" => GradBench.Hello.DIGradientHello(backend)
    )
)

end # module
