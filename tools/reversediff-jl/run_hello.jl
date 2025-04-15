module Hello

using ADTypes: AutoReverseDiff
import ReverseDiff
import GradBench

GradBench.register!(
    "hello", Dict(
        "square" => GradBench.Hello.square,
        "double" => (GradBench.Hello.double, AutoReverseDiff()),
    )
)

end # module
