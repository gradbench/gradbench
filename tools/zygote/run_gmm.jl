module GMM

import Zygote
import GradBench

function unpack(input)
    gmm_input = GradBench.GMM.input_from_json(input)

    d = size(gmm_input.x, 1)
    k = size(gmm_input.means, 2)
    Qs = cat([GradBench.GMM.get_Q(d, gmm_input.icfs[:, ik]) for ik in 1:k]...;
             dims=[3])

    return (gmm_input.alphas, gmm_input.means, Qs, gmm_input.x, gmm_input.wishart)
end

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

Zygote.@adjoint function Base.map(f, args...)
    ys_and_backs = map((args...) -> Zygote._forward(__context__, f, args...), args...)
    ys, backs = unzip(ys_and_backs)
    ys, function (Δ)
        Δf_and_args_zipped = map((f, δ) -> f(δ), backs, Δ)
        Δf_and_args = unzip(Δf_and_args_zipped)
        Δf = reduce(Zygote.accum, Δf_and_args[1])
        (Δf, Δf_and_args[2:end]...)
    end
end

function get_Qs(x, means, icfs)
    d = size(x, 1)
    k = size(means, 2)
    Qs = cat([GradBench.GMM.get_Q(d, icfs[:, ik]) for ik in 1:k]...;
             dims=[3])
    Qs
end

# FIXME: it is very expensive to redo all the input parsing here for
# every run. We absolutely must hoist it out into a "prepare" stage.
function objective(input)
    gmm_input = GradBench.GMM.input_from_json(input)
    Qs = get_Qs(gmm_input.x, gmm_input.means, gmm_input.icfs)
    return GradBench.GMM.objective(gmm_input.alphas, gmm_input.means, Qs, gmm_input.x, gmm_input.wishart)
end

function jacobian(input)
    gmm_input = GradBench.GMM.input_from_json(input)
    function wrap(alpha, means, icfs)
        Qs = get_Qs(gmm_input.x, means, icfs)
        GradBench.GMM.objective(alpha, means, Qs, gmm_input.x, gmm_input.wishart)
    end

    # It would be acceptable to move the massaging of the Jacobian
    # into a separate function that is not timed, but I doubt it
    # matters much.
    (d_alphas, d_means, d_icfs) =
        Zygote.gradient(wrap, gmm_input.alphas, gmm_input.means, gmm_input.icfs)
    vcat(vec(d_alphas), vec(d_means), vec(d_icfs))
end

GradBench.register!("gmm", Dict(
    "objective" => objective,
    "jacobian" => jacobian
))

end
