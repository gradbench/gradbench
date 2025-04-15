import GradBench

include("run_ba.jl")
include("run_gmm.jl")
include("run_hello.jl")
include("run_ht.jl")
include("run_lse.jl")
include("run_lstm.jl")

GradBench.main("zygote")

