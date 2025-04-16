module GMM

using Enzyme
import GradBench

function objective(input)
    gmm_input = GradBench.GMM.input_from_json(input)
    k = size(gmm_input.means, 2)
    d = size(gmm_input.x, 1)
    Qs = GradBench.GMM.get_Qs(gmm_input.icfs, k, d)
    return GradBench.GMM.objective(gmm_input.alphas, gmm_input.means, Qs, gmm_input.x, gmm_input.wishart)
end

function jacobian(input)
    gmm_input = GradBench.GMM.input_from_json(input)
    k = size(gmm_input.means, 2)
    d = size(gmm_input.x, 1)
    Qs = GradBench.GMM.get_Qs(gmm_input.icfs, k, d)

    J =
        Enzyme.gradient(Reverse,
                        GradBench.GMM.objective,
                        gmm_input.alphas, gmm_input.means, Qs,
                        Const(gmm_input.x), Const(gmm_input.wishart))

    GradBench.GMM.pack_J(J, k, d)
end

GradBench.register!("gmm", Dict(
    "objective" => objective,
    "jacobian" => jacobian
))

end # module
