module LSTM

using Enzyme
import GradBench


struct JacobianLSTM <: GradBench.LSTM.AbstractLSTM end
function (::JacobianLSTM)(input)
    dmain_params = Enzyme.make_zero(input.main_params)
    dextra_params = Enzyme.make_zero(input.extra_params)

    # Enzyme.jl is unable to handle the objective function without this.
    Enzyme.API.strictAliasing!(false)

    Enzyme.autodiff(set_runtime_activity(Reverse),
                    GradBench.LSTM.objective,
                    Duplicated(input.main_params, dmain_params),
                    Duplicated(input.extra_params, dextra_params),
                    Const(input.state),
                    Const(input.sequence))
    return [reduce(vcat,dmain_params); reduce(vcat,dextra_params)]
end

GradBench.register!("lstm", Dict(
    "objective" => GradBench.LSTM.ObjectiveLSTM(),
    "jacobian" => JacobianLSTM()
))


end
