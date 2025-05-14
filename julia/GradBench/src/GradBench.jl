module GradBench

import JSON
using ArgParse

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
    func = mod[params["function"]]
    input = params["input"]
    min_runs = input isa Dict ? get(input, "min_runs", 1) : 1
    min_seconds = input isa Dict ? get(input, "min_seconds", 0) : 0
    @assert min_runs > 0

    timings = Any[]

    args, t = measure(preprocess, func, input)
    push!(timings, Dict("name" => "preprocess", "nanoseconds" => t))

    ret, t = measure(func, args...)
    push!(timings, Dict("name" => "warmup", "nanoseconds" => t))

    output, t = measure(postprocess, func, ret)
    push!(timings, Dict("name" => "postprocess", "nanoseconds" => t))

    # Measure
    elapsed_seconds = 0
    i = 1
    while i <= min_runs || elapsed_seconds <= min_seconds
        _, t = measure(func, args...)
        push!(timings, Dict("name" => "evaluate", "nanoseconds" => t))
        elapsed_seconds += t / 1e9
        i += 1
    end

    return Dict("success" => true, "output" => output, "timings" => timings)
end


function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--multithreaded"
        help = "Enable multithreading"
        action = :store_true
    end

    return parse_args(s)
end

OPTIONS = Dict{String,Any}()

function main(tool)
    global OPTIONS
    OPTIONS = parse_commandline()
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
include("benchmarks/llsq.jl")
include("benchmarks/lse.jl")
include("benchmarks/lstm.jl")
include("benchmarks/ode.jl")

end # module GradBench
