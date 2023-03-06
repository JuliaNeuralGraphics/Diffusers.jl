# TODO AdaLayerNorm

struct TransformerBlock{A1, A2, F, L}
    attention_1::A1
    attention_2::A2
    fwd::F
    norm_1::L
    norm_2::L
    norm_3::L

    only_cross_attention::Bool
end
Flux.@functor TransformerBlock

function TransformerBlock(;
    dim::Int, n_heads::Int, head_dim::Int,
    context_dim::Maybe{Int} = nothing, dropout::Real = 0,
    only_cross_attention::Bool = false,
)
    attention_1 = CrossAttention(; # maybe self-attention
        dim, n_heads, head_dim, dropout,
        context_dim=only_cross_attention ? context_dim : nothing)
    attention_2 = isnothing(context_dim) ? nothing : CrossAttention(;
        dim, n_heads, head_dim, dropout, context_dim)

    fwd = FeedForward(; dim, dropout)

    norm_1 = LayerNorm(dim)
    norm_2 = LayerNorm(dim)
    norm_3 = LayerNorm(dim)

    TransformerBlock(
        attention_1, attention_2, fwd, norm_1, norm_2, norm_3,
        only_cross_attention)
end

function (block::TransformerBlock)(
    x::T, context::Maybe{C} = nothing; mask::Maybe{M} = nothing,
) where {
    T <: AbstractArray{Float32, 3},
    C <: AbstractArray{Float32, 3},
    M <: AbstractMatrix{Bool},
}
    a1 = block.attention_1(
        block.norm_1(x), block.only_cross_attention ? context : nothing; mask)
    x = a1 .+ x

    if block.attention_2 ≢ nothing
        a2 = block.attention_2(block.norm_2(x), context; mask)
        x = a2 .+ x
    end

    block.fwd(block.norm_3(x)) .+ x
end
