import GradBench

include("run_hello.jl")
include("run_logsumexp.jl")

GradBench.main("zygote")
