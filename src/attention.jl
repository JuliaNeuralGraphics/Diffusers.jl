"""
Contains both:
Self-attention ≡ AttentionBlock in HGF
Cross-attention ≡ CrossAttention in HGF
"""
struct Attention{Q, K, V, O, L}
    to_q::Q
    to_k::K
    to_v::V
    to_out::O

    norm::L

    context_dim::Maybe{Int}
    n_heads::Int
    scale::Float32
end
Flux.@functor Attention

cross_attention(a::Attention) = !isnothing(a.context_dim)

# NOTE no add kv projection
function Attention(dim::Int;
    bias::Bool,
    head_dim::Int = 64, n_heads::Int = 8,
    n_groups::Maybe{Int} = nothing,
    context_dim::Maybe{Int} = nothing,
    cross_attention_norm::Bool = false,
    scale::Float32 = 1f0,
    dropout::Real = 0,
)
    cross_attention_norm && isnothing(context_dim) && throw(ArgumentError("""
        `context_dim` is `nothing`, but `cross_attention_norm` is `true`.
        Specify either `context_dim` or set `cross_attention_norm` to `false`.
        """))
    !isnothing(n_groups) && !isnothing(context_dim) && throw(ArgumentError("""
        Both `context_dim` and `n_groups` are not `nothing`, but `n_groups`
        are only applicable in self-attention, not cross-attention.
        Either set `n_groups=nothing` or `context_dim=nothing` (self-attention).
        """))

    is_cross_attention = !isnothing(context_dim)
    inner_dim = is_cross_attention ? (head_dim * n_heads) : dim
    ctx_dim = is_cross_attention ? context_dim : dim

    to_q = Dense(dim => inner_dim; bias)
    to_k = Dense(ctx_dim => inner_dim; bias)
    to_v = Dense(ctx_dim => inner_dim; bias)

    to_out = Chain(
        Dense(inner_dim => dim),
        iszero(dropout) ? identity : Dropout(dropout))

    norm = if is_cross_attention
        cross_attention_norm ? LayerNorm(context_dim) : identity
    else
        isnothing(n_groups) ? identity : GroupNorm(dim, n_groups)
    end

    Attention(
        to_q, to_k, to_v, to_out, norm,
        context_dim, n_heads, scale)
end

function (attn::Attention)(
    x::T, context::Maybe{C} = nothing; mask::Maybe{M} = nothing,
) where {
    T <: AbstractArray{Float32, 3},
    C <: AbstractArray{Float32, 3},
    M <: AbstractMatrix{Bool},
}
    residual = x
    _, seq_length, batch = size(x)

    q, k, v = if cross_attention(attn)
        c = attn.norm(context)
        attn.to_q(x), attn.to_k(c), attn.to_v(c)
    else
        x = attn.norm(x)
        attn.to_q(x), attn.to_k(x), attn.to_v(x)
    end

    mask = isnothing(mask) ? nothing :
        reshape(mask, size(mask, 1), 1, 1, size(mask, 2))
    ω, _ = dot_product_attention(q, k, v; mask, nheads=attn.n_heads)

    o = attn.to_out(reshape(ω, :, seq_length, batch))
    cross_attention(attn) && return o

    (o .+ residual) ./ attn.scale
end
