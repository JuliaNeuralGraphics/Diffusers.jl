struct CrossAttention{
    Q <: Dense, K <: Dense, V <: Dense, O <: Chain,
    L <: Union{LayerNorm, typeof(identity)},
}
    to_q::Q
    to_k::K
    to_v::V
    to_out::O

    norm_cross::L

    dim::Int
    context_dim::Int
    head_dim::Int
    n_heads::Int
end
Flux.@functor CrossAttention

# NOTE no add kv projection
function CrossAttention(;
    dim::Int, context_dim::Maybe{Int} = nothing,
    head_dim::Int = 64, n_heads::Int = 8, dropout::Real = 0,
    cross_attention_norm::Bool = false,
)
    inner_dim = head_dim * n_heads
    context_dim = isnothing(context_dim) ? dim : context_dim

    to_q = Dense(dim => inner_dim; bias=false)
    to_k = Dense(context_dim => inner_dim; bias=false)
    to_v = Dense(context_dim => inner_dim; bias=false)

    to_out = Chain(
        Dense(inner_dim => dim),
        iszero(dropout) ? identity : Dropout(dropout))

    norm_cross = cross_attention_norm ?
        LayerNorm(context_dim) : identity

    CrossAttention(
        to_q, to_k, to_v, to_out, norm_cross,
        dim, context_dim, head_dim, n_heads)
end

function (attn::CrossAttention)(
    x::T, context::Maybe{C} = nothing; mask::Maybe{M} = nothing,
) where {
    T <: AbstractArray{Float32, 3},
    C <: AbstractArray{Float32, 3},
    M <: AbstractMatrix{Bool},
}
    _, seq_length, batch = size(x)

    c = isnothing(context) ? x : context
    c = attn.norm_cross(c)

    q = attn.to_q(x)
    k = attn.to_k(c)
    v = attn.to_v(c)

    mask = isnothing(mask) ? nothing :
        reshape(mask, size(mask, 1), 1, 1, size(mask, 2))
    ω, _ = dot_product_attention(q, k, v; mask, nheads=attn.n_heads)

    attn.to_out(reshape(ω, :, seq_length, batch))
end
