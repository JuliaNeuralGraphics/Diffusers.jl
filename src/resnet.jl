struct Downsample2D{C}
    conv::C
    special_padding::Bool
end
Flux.@functor Downsample2D

function Downsample2D(
    channels::Pair{Int, Int}; use_conv::Bool = false, pad::Int = 1,
)
    special_padding = use_conv && pad == 0
    Downsample2D(use_conv ?
        Conv((3, 3), channels; stride=2, pad) :
        MeanPool((2, 2)), special_padding)
end

function (down::Downsample2D)(x::T) where T <: AbstractArray{<:Real, 4}
    down.special_padding && (x = pad_zeros(x, (0, 1, 0, 1, 0, 0, 0, 0));)
    down.conv(x)
end

struct Upsample2D{C}
    conv::C
end
Flux.@functor Upsample2D

function Upsample2D(
    channels::Pair{Int, Int};
    use_conv::Bool = false,
    use_transpose_conv::Bool = false,
    pad::Int = 1,
)
    Upsample2D(if use_transpose_conv
        ConvTranspose((4, 4), channels; stride=2, pad)
    elseif use_conv
        Conv((3, 3), channels; pad)
    else
        identity
    end)
end

function (up::Upsample2D{C})(
    x::T; output_size::Maybe{Tuple{Int, Int}} = nothing,
) where {C, T <: AbstractArray{<:Real, 4}}
    C <: ConvTranspose && (x = up.conv(x);)

    x = isnothing(output_size) ?
        upsample_nearest(x, (2, 2)) :
        upsample_nearest(x; size=output_size)

    C <: Conv && (x = up.conv(x);)
    return x
end

struct ResnetBlock2D{I, O, N, S, E}
    init_proj::I
    out_proj::O
    norm::N
    conv_shortcut::S
    time_emb_proj::E

    scale::Float32
    embedding_scale_shift::Bool
end
Flux.@functor ResnetBlock2D

function ResnetBlock2D(channels::Pair{Int, Int};
    n_groups::Int = 32, n_groups_out::Maybe{Int} = nothing,
    embedding_scale_shift::Bool = false,
    time_emb_channels::Maybe{Int} = 512,
    use_shortcut::Maybe{Bool} = nothing, conv_out_channels::Maybe{Int} = nothing,
    dropout::Real = 0, λ = swish, scale::Float32 = 1f0,
    ϵ::Float32 = 1f-6,
)
    in_channels, out_channels = channels
    n_groups_out = isnothing(n_groups_out) ? n_groups : n_groups_out
    conv_out_channels = isnothing(conv_out_channels) ? out_channels : conv_out_channels

    use_shortcut = isnothing(use_shortcut) ?
        (in_channels != conv_out_channels) : use_shortcut
    time_emb_out_channels = embedding_scale_shift ?
        (out_channels * 2) : out_channels

    # NOTE no up/down
    init_proj = Chain(
        GroupNorm(in_channels, n_groups, λ; ϵ),
        Conv((3, 3), channels; pad=1))
    out_proj = Chain(
        x -> λ.(x),
        iszero(dropout) ? identity : Dropout(dropout),
        Conv((3, 3), out_channels => conv_out_channels; pad=1))

    norm = GroupNorm(out_channels, n_groups_out; ϵ)
    time_emb_proj = isnothing(time_emb_channels) ?
        identity :
        Chain(x -> λ.(x), Dense(time_emb_channels => time_emb_out_channels))

    conv_shortcut = use_shortcut ?
        Conv((1, 1), in_channels => conv_out_channels) : identity

    ResnetBlock2D(
        init_proj, out_proj, norm, conv_shortcut, time_emb_proj,
        scale, embedding_scale_shift)
end

function (block::ResnetBlock2D)(x::T, time_embedding::Maybe{E}) where {
    T <: AbstractArray{<:Real, 4},
    E <: AbstractMatrix{<:Real},
}
    skip, x = x, block.init_proj(x)

    if time_embedding ≢ nothing
        time_embedding = block.time_emb_proj(time_embedding)
        time_embedding = reshape(time_embedding, 1, 1, size(time_embedding)...)
        block.embedding_scale_shift || (x = x .+ time_embedding;)
    end

    x = block.norm(x)

    TI = eltype(x)
    if time_embedding ≢ nothing && block.embedding_scale_shift
        scale, shift = MLUtils.chunk(time_embedding, 2; dims=3)
        x = x .* (one(TI) .+ scale) .+ shift
    end

    (block.out_proj(x) .+ block.conv_shortcut(skip)) ./ TI(block.scale)
end
