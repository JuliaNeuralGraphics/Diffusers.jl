struct TimestepEmbedding{D1, D2}
    linear1::D1
    linear2::D2
end
Flux.@functor TimestepEmbedding

function TimestepEmbedding(in_channels::Int; time_embed_dim::Int)
    TimestepEmbedding(
        Dense(in_channels => time_embed_dim, swish),
        Dense(time_embed_dim => time_embed_dim))
end

function (t::TimestepEmbedding)(x::T) where T <: AbstractMatrix{<:Real}
    tmp = t.linear1(x)
    y = t.linear2(tmp)
    return y
end

struct SinusoidalEmbedding{E}
    emb::E
end
Flux.@functor SinusoidalEmbedding

Flux.trainable(::SinusoidalEmbedding) = ()

function SinusoidalEmbedding(
    dim::Int; max_period::Float32 = 1f4, freq_shift::Int = 1,
)
    half_dim = dim ÷ 2
    γ = log(max_period) / Float32(half_dim - freq_shift)
    ids = collect(UnitRange{Float32}(0, half_dim - 1))
    emb = exp.(ids .* -γ)
    SinusoidalEmbedding(reshape(emb, :, 1))
end

function (emb::SinusoidalEmbedding{E})(timesteps::T) where {
    E <: AbstractMatrix,
    T <: AbstractVector{Int32},
}
    emb = emb.emb .* reshape(timesteps, 1, :)
    cos_emb, sin_emb = cos.(emb), sin.(emb)

    y = cat(cos_emb, sin_emb; dims=1)
    return y
end
