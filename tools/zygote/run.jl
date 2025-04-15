import GradBench

include("run_gmm.jl")
include("run_hello.jl")
include("run_lse.jl")

GradBench.main("zygote")

