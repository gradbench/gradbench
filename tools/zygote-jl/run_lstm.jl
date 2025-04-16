module LSTM

import Zygote
import GradBench

# FIXME: it is very expensive to redo all the input parsing here for
# every run. We absolutely must hoist it out into a "prepare" stage.
function objective(j)
    input = GradBench.LSTM.input_from_json(j)
    return GradBench.LSTM.objective(input.main_params,
                                    input.extra_params,
                                    input.state,
                                    input.sequence)
end

function jacobian(j)
    input = GradBench.LSTM.input_from_json(j)
    function wrap(main_params, extra_params)
        GradBench.LSTM.objective(main_params,
                                 extra_params,
                                 input.state,
                                 input.sequence)
    end

    (d_main_params, d_extra_params) =
        Zygote.gradient(wrap, input.main_params, input.extra_params)
    vcat(vec(d_main_params), vec(d_extra_params))
end

GradBench.register!("lstm", Dict(
    "objective" => objective,
    "jacobian" => jacobian
))


end
