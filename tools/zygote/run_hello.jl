module Hello

import Zygote
import GradBench

function double(x)
    z, = Zygote.gradient(GradBench.Hello.square, x)
    return z
end

GradBench.register!("hello", Dict(
    "square" => GradBench.Hello.square,
    "double" => double
))

end # module
