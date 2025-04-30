module LSTM

import Zygote
import GradBench

struct JacobianLSTM <: GradBench.LSTM.AbstractLSTM end
function (::JacobianLSTM)(input)
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
    "objective" => GradBench.LSTM.ObjectiveLSTM(),
    "jacobian" => JacobianLSTM()
))


end
