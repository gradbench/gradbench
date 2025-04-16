module Hello

using ADTypes: AutoForwardDiff
import ForwardDiff
import GradBench

GradBench.register!(
    "hello", Dict(
        "square" => GradBench.Hello.square,
	"double" => (x)->GradBench.Hello.double(AutoForwardDiff(), x),
    )
)

end # module
