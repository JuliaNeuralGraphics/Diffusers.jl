struct AutoencoderKL{E, D, C1, C2}
    encoder::E
    decoder::D
    quant_conv::C1
    post_quant_conv::C2

    scaling_factor::Float32
end
Flux.@functor AutoencoderKL

function AutoencoderKL(
    channels::Pair{Int, Int};
    latent_channels::Int = 4,
    block_out_channels = (64,),
    n_block_layers::Int = 1,
    λ = swish,
    n_groups::Int = 32,
    scaling_factor::Float32 = 0.18215f0,
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
    AutoencoderKL(encoder, decoder, quant_conv, post_quant_conv, scaling_factor)
end

function encode(kl::AutoencoderKL, x::T) where T <: Union{AbstractArray{Float32, 4}, AbstractArray{Float16, 4}}
    h = kl.encoder(x)
    moments = kl.quant_conv(h)
    DiagonalGaussian(moments)
end

function decode(kl::AutoencoderKL, z::T) where T <: Union{AbstractArray{Float32, 4}, AbstractArray{Float16, 4}}
    h = kl.post_quant_conv(z)
    kl.decoder(h)
end

function (kl::AutoencoderKL)(
    x::T; sample_posterior::Bool = false,
) where T <: Union{AbstractArray{Float32, 4}, AbstractArray{Float16, 4}}
    posterior = encode(kl, x)
    if sample_posterior
        z = sample(posterior)
    else
        z = mode(posterior)
    end
    decode(kl, z)
end

# TODO tiled encode & decode?

# HGF integration.

function AutoencoderKL(model_name::String; state_file::String, config_file::String)
    state, cfg = load_pretrained_model(model_name; state_file, config_file)
    vae = AutoencoderKL(
        cfg["in_channels"] => cfg["out_channels"];
        latent_channels=cfg["latent_channels"],
        block_out_channels=Tuple(cfg["block_out_channels"]),
        n_groups=cfg["norm_num_groups"],
        n_block_layers=cfg["layers_per_block"])
    load_state!(vae, state)
    vae
end
