struct TimestepEmbedding
    linear1::Dense
    linear2::Dense
end

@Flux.functor TimestepEmbedding

function TimestepEmbedding(;in_channels::Int, time_embed_dim::Int)
    TimestepEmbedding(Dense(in_channels => time_embed_dim), 
        Dense(time_embed_dim => time_embed_dim))
end

(t::TimestepEmbedding)(x::AbstractArray{Float32, 2}) = t.linear2(swish(t.linear1(x)))

""" Create sinusoidal timestep embeddings."""
function get_time_embedding(timesteps::AbstractArray{Int64, 1}, embedding_dim::Int)
    (B,) = size(timesteps)
    half_dim = embedding_dim ÷ 2
    # uses 0:159 as time values (not 1:160)
    emb = exp.(-log(10000) * collect(0.0f0:Float32(half_dim-1)) / 160)
    emb = reshape(repeat(emb, B), (size(emb)..., B))
    emb = emb .* timesteps' # row-wise
    emb = cat(cos.(emb), sin.(emb) ; dims=1)
end