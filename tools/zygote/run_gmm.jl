module GMM

import Zygote
import GradBench

Zygote.@adjoint function GradBench.GMM.diagsums(Qs)
    GradBench.GMM.diagsums(Qs),
    function (Δ)
        Δ′ = zero(Qs)
        for (i, δ) in enumerate(Δ)
            for j in 1:size(Qs, 1)
                Δ′[j, j, i] = δ
            end
        end
        (Δ′,)
    end
end

Zygote.@adjoint function GradBench.GMM.expdiags(Qs)
    GradBench.GMM.expdiags(Qs),
    function (Δ)
        Δ′ = zero(Qs)
        Δ′ .= Δ
        for i in 1:size(Qs, 3)
            for j in 1:size(Qs, 1)
                Δ′[j, j, i] *= exp(Qs[j, j, i])
            end
        end
        (Δ′,)
    end
end

# FIXME: it is very expensive to redo all the input parsing here for
# every run. We absolutely must hoist it out into a "prepare" stage.
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

    function wrap(alpha, means, Qs)
        GradBench.GMM.objective(alpha, means, Qs, gmm_input.x, gmm_input.wishart)
    end

    # It would be acceptable to move the massaging of the Jacobian
    # into a separate function that is not timed, but I doubt it
    # matters much.
    J = Zygote.gradient(wrap, gmm_input.alphas, gmm_input.means, Qs)

    GradBench.GMM.pack_J(J, k, d)
end

GradBench.register!("gmm", Dict(
    "objective" => objective,
    "jacobian" => jacobian
))

end
