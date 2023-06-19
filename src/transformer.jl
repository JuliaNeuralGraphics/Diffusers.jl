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
    only_cross_attention && isnothing(context_dim) && throw(ArgumentError("""
        `only_cross_attention` is `true`, but `context_dim` is `nothing`.
        """))

    attention_1 = Attention(dim;
        bias=false, n_heads, head_dim, dropout,
        context_dim=only_cross_attention ? context_dim : dim)
    attention_2 = isnothing(context_dim) ? nothing :
        Attention(dim; bias=false, n_heads, head_dim, dropout, context_dim)

    fwd = FeedForward(; dim, dropout)

    norm_1 = LayerNorm(dim)
    norm_2 = LayerNorm(dim)
    norm_3 = LayerNorm(dim)

    TransformerBlock(
        attention_1, attention_2, fwd,
        norm_1, norm_2, norm_3,
        only_cross_attention)
end

function (block::TransformerBlock)(
    x::T, context::Maybe{C} = nothing; mask::Maybe{M} = nothing,
) where {
    T <: AbstractArray{<:Real, 3},
    C <: AbstractArray{<:Real, 3},
    M <: AbstractMatrix{Bool},
}
    xn = block.norm_1(x)
    a1 = block.attention_1(xn, block.only_cross_attention ? context : xn; mask)
    sync_free!(xn)
    x = a1 .+ x
    sync_free!(a1)

    if block.attention_2 ≢ nothing
        xn = block.norm_2(x)
        a2 = block.attention_2(xn, context; mask)
        sync_free!(xn)
        x = a2 .+ x
        sync_free!(a2)
    end

    xn = block.norm_3(x)
    y = block.fwd(xn) .+ x
    sync_free!(xn, x)
    return y
end

struct Transformer2D{N, P, B}
    norm::N
    proj_in::P
    proj_out::P
    transformer_blocks::B

    use_linear_projection::Bool
end
Flux.@functor Transformer2D

# NOTE only continuous input supported

function Transformer2D(;
    in_channels::Int, n_heads::Int = 16, head_dim::Int = 88,
    n_layers::Int = 1, n_norm_groups::Int = 32,
    use_linear_projection::Bool = false,
    dropout::Real = 0, context_dim::Maybe{Int} = nothing,
)
    inner_dim = n_heads * head_dim

    norm = GroupNorm(in_channels, n_norm_groups)
    proj_in = use_linear_projection ?
        Dense(in_channels => inner_dim) :
        Conv((1, 1), in_channels => inner_dim)
    proj_out = use_linear_projection ?
        Dense(inner_dim => in_channels) :
        Conv((1, 1), inner_dim => in_channels)

    transformer_blocks = Chain([TransformerBlock(;
        dim=inner_dim, n_heads, head_dim, context_dim, dropout)
        for _ in 1:n_layers]...)

    Transformer2D(
        norm, proj_in, proj_out, transformer_blocks,
        use_linear_projection)
end

function (tr::Transformer2D)(x::T, context::Maybe{C} = nothing) where {
    T <: AbstractArray{<:Real, 4}, C <: AbstractArray{<:Real, 3},
}
    width, height, channels, batch = size(x)
    residual = x

    x = tr.norm(x)

    if tr.use_linear_projection
        x = reshape(x, :, channels, batch)
        x_perm = permutedims(x, (2, 1, 3))
        x = tr.proj_in(x_perm)
        sync_free!(x_perm)
    else
        x = tr.proj_in(x)
        x_res = reshape(x, :, size(x, 3), batch)
        x = permutedims(x_res, (2, 1, 3))
        sync_free!(x_res)
    end

    for block in tr.transformer_blocks
        xn = block(x, context)
        sync_free!(x)
        x = xn
    end

    if tr.use_linear_projection
        x_proj = tr.proj_out(x)
        x = permutedims(x_proj, (2, 1, 3))
        sync_free!(x_proj)
        x = reshape(x, width, height, :, batch)
    else
        x = permutedims(x, (2, 1, 3))
        x_res = reshape(x, width, height, :, batch)
        x = tr.proj_out(x_res)
        sync_free!(x_res)
    end

    y = x .+ residual
    sync_free!(x)
    return y
end
