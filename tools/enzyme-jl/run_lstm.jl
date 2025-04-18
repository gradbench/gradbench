module LSTM

using Enzyme
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
    "objective" => objective,
    "jacobian" => jacobian
))


end
