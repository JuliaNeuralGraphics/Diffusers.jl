struct AuotoencoderKL{E, D, C1, C2}
    encoder::E
    decoder::D
    quant_conv::C1
    post_quant_conv::C2
end
Flux.@functor AutoencoderKL

function AutoencoderKL(
    channels::Pair{Int, Int};
    latent_channels::Int = 4,
    block_out_channels = (64,),
    n_block_layers::Int = 1,
    λ = swish,
    n_groups::Int = 32,
    sample_size::Int = 32,
    scaling_factor::Real = 0.18215,
)
    channels_in, channels_out = channels
    encoder = Encoder(channels_in => latent_channels;
        block_out_channels, n_block_layers,
        n_groups, λ, double_z=true)
    decoder = Decoder(latent_channels => channels_out;
        block_out_channels, n_block_layers,
        n_groups, λ)

    quant_conv = Conv((1, 1), (2 * latent_channels) => (2 * latent_channels))
    post_quant_conv = Conv((1, 1), latent_channels => latent_channels)

    # TODO tiling & slicing

    AutoencoderKL(encoder, decoder, quant_conv, post_quant_conv)
end

# TODO encode & decode
# TODO forward
