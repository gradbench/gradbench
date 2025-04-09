module Hello

using ADTypes: AutoZygote
import Zygote
import GradBench

GradBench.register!("hello", Dict(
    "square" => GradBench.Hello.square,
    "double" => GradBench.Hello.double,
    "backend" => AutoZygote(),
))

end # module
