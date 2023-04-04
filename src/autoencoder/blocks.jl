struct Encoder{C1, C2, N, D, M}
    conv_in::C1
    conv_out::C2
    norm::N
    down_blocks::D
    mid_block::M
end
Flux.@functor Encoder

function Encoder(
    channels::Pair{Int, Int};
    block_out_channels = (64,),
    n_block_layers::Int = 2,
    n_groups::Int = 32,
    λ = swish,
    double_z::Bool = true,
)
    channels_in, channels_out = channels
    output_channels = block_out_channels[1]
    conv_in = Conv((3, 3), channels_in => output_channels; pad=1)

    down_blocks = []
    for (i, boc) in enumerate(block_out_channels)
        input_channels, output_channels = output_channels, boc
        is_final = i == length(block_out_channels)

        push!(down_blocks, SamplerBlock2D{Downsample2D}(
            input_channels => output_channels;
            n_groups, n_layers=n_block_layers,
            sampler_pad=0, add_sampler=!is_final, λ))
    end

    mid_block = MidBlock2D(
        block_out_channels[end];
        time_emb_channels=nothing,
        n_heads=1, n_groups, λ,
        embedding_scale_shift=false)

    norm = GroupNorm(output_channels, n_groups, λ)
    channels_out = double_z ? (channels_out * 2) : channels_out
    conv_out = Conv((3, 3), output_channels => channels_out; pad=1)

    Encoder(conv_in, conv_out, norm, Chain(down_blocks...), mid_block)
end

function (enc::Encoder)(x::T) where T <: Union{AbstractArray{Float32, 4}, AbstractArray{Float16, 4}}
    x = enc.conv_in(x)
    x = enc.down_blocks(x)
    x = enc.mid_block(x)
    x = enc.norm(x)
    enc.conv_out(x)
end

struct Decoder{C1, C2, N, U, M}
    conv_in::C1
    conv_out::C2
    norm::N
    up_blocks::U
    mid_block::M
end
Flux.@functor Decoder

function Decoder(
    channels::Pair{Int, Int};
    block_out_channels = (64,),
    n_block_layers::Int = 2,
    n_groups::Int = 32,
    λ = swish,
)
    channels_in, channels_out = channels
    block_out_channels = reverse(block_out_channels)
    output_channels = block_out_channels[1]
    conv_in = Conv((3, 3), channels_in => output_channels; pad=1)

    up_blocks = []
    for (i, boc) in enumerate(block_out_channels)
        input_channels, output_channels = output_channels, boc
        is_final = i == length(block_out_channels)

        push!(up_blocks, SamplerBlock2D{Upsample2D}(
            input_channels => output_channels;
            n_groups, n_layers=n_block_layers + 1,
            add_sampler=!is_final, λ))
    end

    mid_block = MidBlock2D(
        block_out_channels[1];
        time_emb_channels=nothing,
        n_heads=1, n_groups, λ,
        embedding_scale_shift=false)

    norm = GroupNorm(output_channels, n_groups, λ)
    conv_out = Conv((3, 3), output_channels => channels_out; pad=1)

    Decoder(conv_in, conv_out, norm, Chain(up_blocks...), mid_block)
end

function (dec::Decoder)(x::T) where T <: Union{AbstractArray{Float32, 4}, AbstractArray{Float16, 4}}
    x = dec.conv_in(x)
    x = dec.mid_block(x)
    x = dec.up_blocks(x)
    dec.conv_out(dec.norm(x))
end

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
) where T <: Union{AbstractArray{Float32, 4}, AbstractArray{Float16, 4}}
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

mode(dg::DiagonalGaussian) = dg.μ
