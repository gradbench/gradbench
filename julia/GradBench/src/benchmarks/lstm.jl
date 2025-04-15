# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
#
## Based on: https://github.com/microsoft/ADBench/blob/38cb7931303a830c3700ca36ba9520868327ac87/src/julia/modules/Zygote/ZygoteLSTM.jl

module LSTM

export LSTMInput, input_from_json, objective

struct LSTMInput
    main_params::Matrix{Float64}
    extra_params::Matrix{Float64}
    state::Matrix{Float64}
    sequence::Matrix{Float64}
end

function input_from_json(j)
    main_params = reduce(hcat, convert(Vector{Vector{Float64}}, j["main_params"]))
    extra_params = reduce(hcat, convert(Vector{Vector{Float64}}, j["extra_params"]))
    state = reduce(hcat, convert(Vector{Vector{Float64}}, j["state"]))
    sequence = reduce(hcat, convert(Vector{Vector{Float64}}, j["sequence"]))
    return LSTMInput(main_params, extra_params, state, sequence)
end

function sigmoid(x)
    1. / (1. + exp(-x))
end

function lstmmodel(weight::T1, bias::T2, hidden::T3, cell::T4, input::T5)::Tuple{Vector{Float64}, Vector{Float64}} where {T1<:AbstractVector{Float64}, T2<:AbstractVector{Float64}, T3<:AbstractVector{Float64}, T4<:AbstractVector{Float64}, T5<:AbstractVector{Float64}}
    hsize = size(hidden, 1)
    forget = sigmoid.(input .* view(weight, 1:hsize) .+ view(bias, 1:hsize))
    ingate = sigmoid.(hidden .* view(weight, hsize+1:2hsize) .+ view(bias, hsize+1:2hsize))
    outgate = sigmoid.(input .* view(weight, 2hsize+1:3hsize) .+ view(bias, 2hsize+1:3hsize))
    change = tanh.(hidden .* view(weight, 3hsize+1:4hsize) .+ view(bias, 3hsize+1:4hsize))

    cell2 = cell .* forget .+ ingate .* change
    hidden2 = outgate .* tanh.(cell2)
    hidden2, cell2
end

function lstmpredict(main_params::Matrix{Float64}, extra_params::Matrix{Float64}, state::Matrix{Float64}, input::T)::Tuple{Vector{Float64}, Matrix{Float64}} where {T<:AbstractVector{Float64}}
    x1 = input .* view(extra_params, :, 1)
    x = view(x1, :)
    lenstate = size(state, 2)
    s2 = Matrix{Float64}(undef, size(input, 1), 0)
    for i ∈ 1:2:lenstate
        h, c = lstmmodel(view(main_params, :, i), view(main_params, :, i + 1), view(state, :, i), view(state, :, i + 1), x)
        x = h
        # Zygote does not support mutating arrays
        # TODO: rewrite as array comprehension - initial attempts
        # were not successfully differentiated by Zygote
        s2 = hcat(s2, h, c)
    end
    (x .* view(extra_params, :, 2) .+ view(extra_params, :, 3), s2)
end

function objective(main_params::Matrix{Float64},
                   extra_params::Matrix{Float64},
                   state::Matrix{Float64},
                   sequence::Matrix{Float64})::Float64
    total = 0.
    count = 0
    input = view(sequence, :, 1)
    b, c = size(sequence)
    curstate = state
    for t ∈ 1:c-1
        ypred, curstate = lstmpredict(main_params, extra_params, curstate, input)
        ynorm = ypred .- log(sum(exp, ypred) + 2.)
        ygold = view(sequence, :, t + 1)
        total += sum(ygold .* ynorm)
        count += b
        input = ygold
    end
    -total / count
end

end
