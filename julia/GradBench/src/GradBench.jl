module GradBench

import JSON

"""
    Experiment

Defines a measurment experiment.

## Interface
- `args = preprocess(::Experiment, message)`
- `ret = (::Experiment)(args...)`
- Optional: `postprocess(::Experiment, ret)`
"""
abstract type Experiment end

function preprocess(::Experiment, message)
    return message
end

function postprocess(::Experiment, ret)
    return ret
end

const DISPATCH_TABLE = Dict{String,Dict{String,Experiment}}()

function register!(mod, experiments)
    if haskey(DISPATCH_TABLE, mod)
        error("mod $mod is already registered")
    end
    DISPATCH_TABLE[mod] = experiments
    return
end

# Avoid measuring dispatch overhead
function measure(func::F, args...) where {F}
    start = time_ns()
    ret = func(args...)
    done = time_ns()
    return ret, done - start
end

function run(params)
    mod = DISPATCH_TABLE[params["module"]]
    experiment = mod[params["function"]]
    min_runs = get(params, "min_runs", 1)
    min_seconds = get(params, "min_seconds", 0)
    @assert min_runs > 0

    timings = Any[]

    args, t = measure(preprocess, experiment, params["input"])
    push!(timings, Dict("name" => "preprocess", "nanoseconds" => t))

    ret, t = measure(experiment, args...)
    push!(timings, Dict("name" => "warmup", "nanoseconds" => t))

    output, t = measure(postprocess, experiment, ret)
    push!(timings, Dict("name" => "postprocess", "nanoseconds" => t))

    # Measure
    elapsed_seconds = 0
    i = 1
    while i <= min_runs || elapsed_seconds <= min_seconds
        _, t = measure(experiment, args...)
        push!(timings, Dict("name" => "evaluate", "nanoseconds" => t))
        elapsed_seconds += t / 1e9
        i += 1
    end

    return Dict("success" => true, "output" => output, "timings" => timings)
end

function main(tool)
    while !eof(stdin)
        message = JSON.parse(readline(stdin))
        response = Dict()
        if message["kind"] == "start"
            response["tool"] = tool
        elseif message["kind"] == "evaluate"
            response = run(message)
        elseif message["kind"] == "define"
            response["success"] = haskey(DISPATCH_TABLE, message["module"])
        end
        response["id"] = message["id"]
        println(JSON.json(response))
    end
    return
end

include("benchmarks/ba.jl")
include("benchmarks/det.jl")
include("benchmarks/gmm.jl")
include("benchmarks/hello.jl")
include("benchmarks/ht.jl")
include("benchmarks/ode.jl")
include("benchmarks/lse.jl")
include("benchmarks/lstm.jl")

end # module GradBench
