struct CrossAttnDownBlock2D{R, A, D}
    resnets::R
    attentions::A
    downsamplers::D
end
Flux.@functor CrossAttnDownBlock2D

function CrossAttnDownBlock2D(;
    in_channels::Int, out_channels::Int, time_emb_channels::Int,
    dropout::Real = 0, n_layers::Int = 1, resnet_time_scale_shift::Bool = false,
    resnet_λ = swish, resnet_groups::Int = 32, attn_n_heads::Int = 1,
    down_padding::Int = 1, context_dim::Int = 1280, add_downsample::Bool = true,
    use_linear_projection::Bool = false,
)
    resnets = Chain([ResnetBlock2D(
        (i == 1 ? in_channels : out_channels) => out_channels;
        λ = resnet_λ, time_emb_channels, n_groups=resnet_groups, dropout,
        embedding_scale_shift=resnet_time_scale_shift) for i in 1:n_layers]...)

    attentions = Chain([Transformer2D(; in_channels=out_channels, n_heads=attn_n_heads,
        dropout, head_dim=out_channels÷attn_n_heads, n_norm_groups=resnet_groups,
        use_linear_projection, context_dim) for i in 1:n_layers]...)

    downsamplers = isnothing(add_downsample) ? identity : Chain(
        Conv((3, 3), in_channels => out_channels; stride=2, pad=down_padding))

    CrossAttnDownBlock2D(resnets, attentions, downsamplers)
end

function (cattn::CrossAttnDownBlock2D)(
    x::T, time_emb::Maybe{E} = nothing, context::Maybe{C} = nothing,
) where {
    T <: AbstractArray{Float32, 4},
    E <: AbstractArray{Float32, 2},
    C <: AbstractArray{Float32, 3},
}
    # Note: attention_mask is not used
    for (resnet, attn) in zip(cattn.resnets, cattn.attentions)
        x = resnet(x, time_emb)
        x = attn(x, context)
    end
    x = cattn.downsamplers(x)
    return x
end

# UNetMidBlock2DCrossAttn in diffusers.py
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
    T <: AbstractArray{Float32, 4},
    E <: AbstractArray{Float32, 2},
    C <: AbstractArray{Float32, 3},
}
    x = mid.resnets[1](x, time_emb)
    for (resnet, attn) in zip(mid.resnets[2:end], mid.attentions)
        x = attn(x, context)
        x = resnet(x, time_emb)
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

function (block::SamplerBlock2D)(x::T) where T <: AbstractArray{Float32, 4}
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
)
    resnets = [ResnetBlock2D(
        channels => channels; time_emb_channels, scale, embedding_scale_shift,
        n_groups, dropout, λ)
        for _ in 1:(n_layers + 1)]
    attentions = add_attention ?
        Chain([
            Attention(channels; bias=true, n_heads, n_groups, scale)
            for _ in 1:n_layers]...) : nothing
    MidBlock2D(Chain(resnets...), attentions)
end

function (mb::MidBlock2D{R, A})(
    x::T, time_embedding::Maybe{E} = nothing,
) where {
    R, A, T <: AbstractArray{Float32, 4},
    E <: AbstractMatrix{Float32},
}
    x = mb.resnets[1](x, time_embedding)
    for i in 2:length(mb.resnets)
        if A <: Nothing
            width, height, channels, batch = size(x)
            x = mb.attentions[i - 1](reshape(x, :, channels, batch))
            x = reshape(x, width, height, channels, batch)
        end
        x = mb.resnets[i](x, time_embedding)
    end
    x
end

struct UpBlock2D{R}
    resnets::R
    sampler::Maybe{Upsample2D}  # this can also be identity
end
Flux.@functor UpBlock2D

function UpBlock2D(
    channels::Pair{Int, Int}, prev_out_channel::Int, temb_channels::Int;
    n_layers::Int = 1, n_groups::Int = 32,
    dropout::Real = 0
)
    in_channels, out_channels = channels
    resnets = []

    for i in 1:n_layers
        res_skip_channels = (i == n_layers - 1) ? in_channels : out_channels
        res_in_channels = (i == 0) ? prev_out_channel : out_channels
        push!(resnets, ResnetBlock2D(
            (res_in_channels + res_skip_channels) => out_channels;
            time_emb_channels=temb_channels, embedding_scale_shift,
            n_groups, dropout, λ))
    end
    resnets = Chain(resnets...)

    sampler = add_sampler ?
        Upsample2D(out_channels=>out_channels; use_conv=true, pad=sampler_pad) : nothing
    UpBlock2D(resnets, sampler)
end

function (block::UpBlock2D)(x::T, skip_x::NTuple, temb::E) where {
    T <: AbstractArray{Float32, 4}, E <: AbstractArray{Float32, 2},
}
    skip_x = reverse(skip_x)
    for (i, rn) in enumerate(block.resnets)
        skip = skip_x[i]
        x = cat(x, skip; dims=3)
        x = rn(x, temb)
    end
    if ! isnothing(block.sampler) x = block.sampler(x) end
    return x
end


struct DownBlock2D{R}
    resnets::R
    sampler::Maybe{Downsample2D} # harish: not identity; what type then?
end
Flux.@functor DownBlock2D

function DownBlock2D(
    channels::Pair{Int, Int}, temb_channels::Int; n_layers::Int = 1,
    n_groups::Int = 32, embedding_scale_shift::Bool = false,
    add_sampler::Bool = true, sampler_pad::Int = 1, λ = swish, dropout::Real = 0,
)
    in_channels, out_channels = channels
    resnets = []

    for i in 1:n_layers
        in_channels = (i == 0) ? in_channels : out_channels
        push!(resnets, ResnetBlock2D(
            in_channels => out_channels; time_emb_channels=temb_channels,
            embedding_scale_shift,
            n_groups, dropout, λ))
    end
    resnets = Chain(resnets...)
    sampler = add_sampler ?
        Downsample2D(out_channels=>out_channels; use_conv=true, pad=sampler_pad) : nothing
    DownBlock2D(resnets, sampler)
end

function (block::DownBlock2D)(x::T, temb::E) where {
    T <: AbstractArray{Float32, 4}, E <: AbstractArray{Float32, 2},
}
    for rn in block.resnets
        x = rn(x, temb)
    end
    if ! isnothing(block.sampler) x = block.sampler(x) end
    return x
end
