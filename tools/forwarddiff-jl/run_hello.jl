module Hello

using ADTypes: AutoForwardDiff
import ForwardDiff
import GradBench

GradBench.register!(
    "hello", Dict(
        "square" => GradBench.Hello.square,
        "double" => (GradBench.Hello.double, AutoForwardDiff()),
    )
)

end # module
