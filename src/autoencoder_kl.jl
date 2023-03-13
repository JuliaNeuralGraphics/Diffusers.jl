struct DiagonalGaussian{M, S, N, L}
    μ::M
    σ::S
    ν::N
    log_σ::L
end

function DiagonalGaussian(θ)
    μ, log_σ = MLUtils.chunk(θ, 2; dims=ndims(θ) - 1) # Slice channel dim.
    clamp!(log_σ, -30f0, 20f0)
    σ = exp.(0.5f0 .* log_σ)
    ν = exp.(log_σ)
    DiagonalGaussian(μ, σ, ν, log_σ)
end

function sample(dg::DiagonalGaussian{M, S, N, L}) where {M, S, N, L}
    ξ = randn(eltype(M), size(dg.μ)) # TODO generate on device
    dg.μ .+ dg.σ .* ξ
end

# Kullback–Leibler divergence.
function kl(
    dg::DiagonalGaussian{T}, other::Maybe{DiagonalGaussian{T}} = nothing,
) where T <: AbstractArray{Float32, 4}
    dims = (1, 2, 3)
    0.5f0 .* (isnothing(other) ?
        sum(dg.μ.^2 .+ dg.ν .- dg.log_σ .- 1f0; dims) :
        sum(
            (dg.μ .- other.μ).^2 ./ other.ν .+
            dg.ν ./ other.ν .-
            dg.log_σ .+ other.log_σ .- 1f0; dims))
end

# Negative Log Likelihood.
function nll(dg::DiagonalGaussian, x; dims = (1, 2, 3))
    0.5f0 .* sum(log(2f0 .* π) .+ dg.log_σ .+ (x .- dg.μ).^2 ./ dg.ν; dims)
end
