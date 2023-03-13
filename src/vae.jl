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

        push!(down_blocks, SamplerBlock2D(
            input_channels => output_channels, Downsample2D;
            n_groups, n_layers=n_block_layers,
            add_sampler=!is_final, sampler_pad=0, λ))
    end

    mid_block = identity # TODO UNetMidBlock2D

    norm = GroupNorm(output_channels, n_groups, λ)
    channels_out = double_z ? (channels_out * 2) : channels_out
    conv_out = Conv((3, 3), output_channels => channels_out; pad=1)

    Encoder(conv_in, conv_out, norm, Chain(down_blocks...), mid_block)
end

function (enc::Encoder)(x::T) where T <: AbstractArray{Float32, 4}
    x = enc.conv_in(x)
    x = enc.down_blocks(x)
    x = enc.mid_block(x)
    enc.conv_out(enc.norm(x))
end

struct Decoder{C1, C2, N, U, M}
    conv_in::C1
    conv_out::C2
    norm::N,
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

        push!(up_blocks, SamplerBlock2D(
            input_channels => output_channels, Upsample2D;
            n_groups, n_layers=n_block_layers + 1,
            add_sampler=!is_final, λ))
    end

    mid_block = identity # TODO UnetMidBlock2D

    norm = GroupNorm(output_channels, n_groups, λ)
    conv_out = Conv((3, 3), output_channels => channels_out; pad=1)

    Decoder(conv_in, conv_out, norm, Chain(up_blocks...), mid_block)
end

function (dec::Decoder)(x::T) where T <: AbstractArray{Float32, 4}
    x = dec.conv_in(x)
    x = dec.mid_block(x)
    x = dec.up_blocks(x)
    dec.conv_out(dec.norm(x))
end

# TODO load_state! for both
