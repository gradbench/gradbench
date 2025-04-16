module GMM

using Enzyme
import GradBench

function get_Qs(x, means, icfs)
    d = size(x, 1)
    k = size(means, 2)
    Qs = cat([GradBench.GMM.get_Q(d, icfs[:, ik]) for ik in 1:k]...;
             dims=[3])
    Qs
end

function objective(input)
    gmm_input = GradBench.GMM.input_from_json(input)
    Qs = get_Qs(gmm_input.x, gmm_input.means, gmm_input.icfs)
    return GradBench.GMM.objective(gmm_input.alphas, gmm_input.means, Qs, gmm_input.x, gmm_input.wishart)
end

function jacobian(input)
    gmm_input = GradBench.GMM.input_from_json(input)

    function wrap(alpha, means, icfs, x, wishart)
        Qs = get_Qs(x, means, icfs)
        GradBench.GMM.objective(alpha, means, Qs, x, wishart)
    end

    (d_alphas, d_means, d_icfs) =
        Enzyme.gradient(Reverse,
                        wrap,
                        gmm_input.alphas, gmm_input.means, gmm_input.icfs,
                        Const(gmm_input.x), Const(gmm_input.wishart))

    vcat(vec(d_alphas), vec(d_means), vec(d_icfs))
end

GradBench.register!("gmm", Dict(
    "objective" => objective,
    "jacobian" => jacobian
))

end # module
