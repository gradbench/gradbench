import GradBench

include("run_det.jl")
include("run_hello.jl")
include("run_ode.jl")

GradBench.main("mooncake-jl")
