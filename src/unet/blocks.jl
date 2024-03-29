struct CrossAttnDownBlock2D{R, A, D}
    resnets::R
    attentions::A
    downsamplers::D
end
Flux.@functor CrossAttnDownBlock2D

function CrossAttnDownBlock2D(
    channels::Pair{Int, Int}; time_emb_channels::Int, dropout::Real = 0,
    n_layers::Int = 1, embedding_scale_shift::Bool = false,
    λ = swish, n_groups::Int = 32, n_heads::Int = 1,
    context_dim::Int = 1280, use_linear_projection::Bool = false,
    add_downsample::Bool = true, pad::Int = 1,
)
    in_channels, out_channels = channels
    head_dim = out_channels ÷ n_heads

    resnets = ([
        ResnetBlock2D(
            (i == 1 ? in_channels : out_channels) => out_channels;
            λ, time_emb_channels, n_groups, dropout, embedding_scale_shift)
        for i in 1:n_layers]...,)
    attentions = ([
        Transformer2D(;
            in_channels=out_channels, n_heads, dropout, head_dim,
            n_norm_groups=n_groups, use_linear_projection, context_dim)
        for i in 1:n_layers]...,)

    downsamplers = add_downsample ?
        Conv((3, 3), out_channels => out_channels; stride=2, pad) :
        identity

    CrossAttnDownBlock2D(resnets, attentions, downsamplers)
end

has_sampler(::CrossAttnDownBlock2D{R, A, D}) where {R, A, D} = !(D <: typeof(identity))

function (cattn::CrossAttnDownBlock2D)(
    x::T, time_emb::Maybe{E} = nothing, context::Maybe{C} = nothing,
) where {
    T <: AbstractArray{<:Real, 4},
    E <: AbstractArray{<:Real, 2},
    C <: AbstractArray{<:Real, 3},
}
    function _chain(resnets::Tuple, attentions::Tuple, h)
        tmp = first(resnets)(h, time_emb)
        h = first(attentions)(tmp, context)
        (h, _chain(Base.tail(resnets), Base.tail(attentions), h)...)
    end
    _chain(::Tuple{}, ::Tuple{}, _) = ()

    states = _chain(cattn.resnets, cattn.attentions, x)
    x = states[end]

    has_sampler(cattn) || return x, states
    x = cattn.downsamplers(x)
    x, (states..., x)
end

struct DownBlock2D{R, S}
    resnets::R
    sampler::S
end
Flux.@functor DownBlock2D

function DownBlock2D(
    channels::Pair{Int, Int}, time_emb_channels::Int; n_layers::Int = 1,
    n_groups::Int = 32, embedding_scale_shift::Bool = false,
    add_downsample::Bool = true, pad::Int = 1, λ = swish, dropout::Real = 0,
)
    in_channels, out_channels = channels
    resnets = [
        ResnetBlock2D(
            (i == 1 ? in_channels : out_channels) => out_channels;
            time_emb_channels, embedding_scale_shift, n_groups, dropout, λ)
        for i in 1:n_layers]
    sampler = add_downsample ?
        Downsample2D(out_channels => out_channels; use_conv=true, pad) :
        identity
    DownBlock2D((resnets...,), sampler)
end

has_sampler(::DownBlock2D{R, S}) where {R, S} = !(S <: typeof(identity))

function (block::DownBlock2D)(x::T, temb::E) where {
    T <: AbstractArray{<:Real, 4},
    E <: AbstractArray{<:Real, 2},
}
    function _chain(blocks::Tuple, h)
        h = first(blocks)(h, temb)
        (h, _chain(Base.tail(blocks), h)...)
    end
    _chain(::Tuple{}, _) = ()

    states = _chain(block.resnets, x)
    x = states[end]

    has_sampler(block) || return x, states
    x = block.sampler(x)
    x, (states..., x)
end

struct CrossAttnMidBlock2D{R, A}
    resnets::R
    attentions::A
end
Flux.@functor CrossAttnMidBlock2D

function CrossAttnMidBlock2D(;
    in_channels::Int, time_emb_channels::Maybe{Int},
    dropout::Real = 0,
    n_layers::Int = 1,
    embedding_scale_shift::Bool = false,
    λ = swish,
    n_groups::Int = 32,
    n_heads::Int = 1,
    context_dim::Int = 1280,
    use_linear_projection::Bool = false,
)
    resnets = [ResnetBlock2D(in_channels => in_channels;
        n_groups, embedding_scale_shift, time_emb_channels,
        dropout, λ)]
    attentions = []

    for i in 1:n_layers
        push!(attentions, Transformer2D(;
            in_channels, n_heads, context_dim,
            dropout, use_linear_projection, head_dim=in_channels ÷ n_heads,
            n_norm_groups=n_groups))
        push!(resnets, ResnetBlock2D(in_channels => in_channels;
            n_groups, embedding_scale_shift, time_emb_channels,
            dropout, λ))
    end
    CrossAttnMidBlock2D(Chain(resnets...), Chain(attentions...))
end

function (mid::CrossAttnMidBlock2D)(
    x::T, time_emb::Maybe{E} = nothing, context::Maybe{C} = nothing,
) where {
    T <: AbstractArray{<:Real, 4},
    E <: AbstractArray{<:Real, 2},
    C <: AbstractArray{<:Real, 3},
}
    x = mid.resnets[1](x, time_emb)
    for (resnet, attn) in zip(mid.resnets[2:end], mid.attentions)
        tmp = attn(x, context)
        x = resnet(tmp, time_emb)
    end
    x
end

"""
When `sampler` <: `Downsample2D` ≡ DownEncoderBlock2D from HGF.
When `sampler` <: `Upsample2D` ≡ UpDecoderBlock2D from HGF.
"""
struct SamplerBlock2D{R, S}
    resnets::R
    sampler::S
end
Flux.@functor SamplerBlock2D

function SamplerBlock2D{S}(
    channels::Pair{Int, Int};
    n_layers::Int = 1,
    n_groups::Int = 32,
    embedding_scale_shift::Bool = false,
    add_sampler::Bool = true,
    sampler_pad::Int = 1,
    λ = swish,
    dropout::Real = 0,
) where S <: Union{Downsample2D, Upsample2D}
    out_channels = channels[2] => channels[2]
    resnets = Chain([
        ResnetBlock2D(i == 1 ? channels : out_channels;
            time_emb_channels=nothing,
            embedding_scale_shift,
            n_groups, dropout, λ)
        for i in 1:n_layers]...)

    sampler = add_sampler ?
        S(out_channels; use_conv=true, pad=sampler_pad) : identity
    SamplerBlock2D(resnets, sampler)
end

function (block::SamplerBlock2D)(x::T) where T <: AbstractArray{<:Real, 4}
    for rn in block.resnets
        x = rn(x, nothing)
    end
    block.sampler(x)
end

# TODO all mid blocks can share this
"""
Generic UNet middle block.

In HGF diffusers it corresponds to `UNetMidBlock2D<attention-layer-name>`.
"""
struct MidBlock2D{R, A}
    resnets::R
    attentions::A
end
Flux.@functor MidBlock2D

function MidBlock2D(
    channels::Int;
    n_layers::Int = 1,
    n_groups::Int = 32,
    time_emb_channels::Maybe{Int},
    embedding_scale_shift::Bool = false,
    add_attention::Bool = true,
    dropout::Real = 0,
    λ = swish,
    scale::Float32 = 1f0,
    n_heads::Int = 1,
    ϵ::Float32 = 1f-6,
)
    resnets = [ResnetBlock2D(
        channels => channels; time_emb_channels, scale, embedding_scale_shift,
        n_groups, dropout, λ)
        for _ in 1:(n_layers + 1)]
    attentions = add_attention ?
        Chain([
            Attention(channels; bias=true, n_heads, n_groups, scale, ϵ)
            for _ in 1:n_layers]...) : nothing
    MidBlock2D(Chain(resnets...), attentions)
end

function (mb::MidBlock2D{R, A})(
    x::T, time_embedding::Maybe{E} = nothing,
) where {
    R, A,
    T <: AbstractArray{<:Real, 4},
    E <: AbstractMatrix{<:Real},
}
    x = mb.resnets[1](x, time_embedding)
    for i in 2:length(mb.resnets)
        if !(A <: Nothing)
            width, height, channels, batch = size(x)
            x = mb.attentions[i - 1](reshape(x, :, channels, batch))
            x = reshape(x, width, height, channels, batch)
        end
        x = mb.resnets[i](x, time_embedding)
    end
    x
end

struct CrossAttnUpBlock2D{R, A, S}
    resnets::R
    attentions::A
    sampler::S
end
Flux.@functor CrossAttnUpBlock2D

has_sampler(::CrossAttnUpBlock2D{R, A, S}) where {R, A, S} = !(S <: typeof(identity))

function CrossAttnUpBlock2D(
    channels::Pair{Int, Int}, prev_out_channel::Int, time_emb_channels::Int;
    dropout::Real = 0, n_layers::Int=1, resnet_time_scale_shift::Bool = false,
    resnet_λ = swish, n_groups::Int=32, attn_n_heads::Int=1, add_upsample::Bool=true,
    context_dim::Int=1280, use_linear_projection::Bool=false)

    in_channels, out_channels = channels
    resnets = []
    attentions = []

    for i in 1:n_layers
        res_skip_channels = (i == n_layers) ? in_channels : out_channels
        res_in_channels = (i == 1) ? prev_out_channel : out_channels

        push!(resnets, ResnetBlock2D(
            (res_in_channels + res_skip_channels) => out_channels;
            time_emb_channels, embedding_scale_shift=resnet_time_scale_shift,
            n_groups, dropout, λ=resnet_λ))
        push!(attentions, Transformer2D(; in_channels=out_channels, n_heads=attn_n_heads,
            dropout, head_dim=out_channels÷attn_n_heads, n_norm_groups=n_groups,
            use_linear_projection, context_dim))
    end
    sampler = add_upsample ?
        Upsample2D(out_channels=>out_channels; use_conv=true) : identity
    CrossAttnUpBlock2D(Chain(resnets...), Chain(attentions...), sampler)
end

function (block::CrossAttnUpBlock2D)(
    x::T, skips, temb::Maybe{E} = nothing, context::Maybe{C} = nothing
) where {
    T <: AbstractArray{<:Real, 4},
    E <: AbstractArray{<:Real, 2},
    C <: AbstractArray{<:Real, 3},
}
    for (rn, attn) in zip(block.resnets, block.attentions)
        skip, skips = first(skips), Base.tail(skips)
        x = cat(x, skip; dims=3)
        x = rn(x, temb)
        x = attn(x, context)
    end
    block.sampler(x), skips
end

struct UpBlock2D{R, S}
    resnets::R
    sampler::S
end
Flux.@functor UpBlock2D

has_sampler(::UpBlock2D{R, S}) where {R, S} = !(S <: typeof(identity))

function UpBlock2D(
    channels::Pair{Int, Int}, prev_out_channel::Int, time_emb_channels::Int;
    n_layers::Int = 1, n_groups::Int = 32,
    add_upsample::Bool = true, sampler_pad::Int = 1, λ = swish,
    embedding_scale_shift::Bool = false, dropout::Real = 0
)
    in_channels, out_channels = channels

    resnets = []
    for i in 1:n_layers
        res_skip_channels = (i == n_layers - 1) ? in_channels : out_channels
        res_in_channels = (i == 0) ? prev_out_channel : out_channels
        push!(resnets, ResnetBlock2D(
            (res_in_channels + res_skip_channels) => out_channels;
            time_emb_channels, embedding_scale_shift, n_groups, dropout, λ))
    end

    sampler = add_upsample ?
        Upsample2D(out_channels=>out_channels; use_conv=true, pad=sampler_pad) :
        identity
    UpBlock2D(Chain(resnets...), sampler)
end

function (block::UpBlock2D)(x::T, skips, temb::E) where {
    T <: AbstractArray{<:Real, 4},
    E <: AbstractArray{<:Real, 2},
}
    for block in block.resnets
        skip, skips = first(skips), Base.tail(skips)
        tmp = cat(x, skip; dims=3)
        x = block(tmp, temb)
    end
    y = block.sampler(x)
    y, skips
end
