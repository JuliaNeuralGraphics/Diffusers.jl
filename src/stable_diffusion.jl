struct StableDiffusion{V, T, K, U, S}
    vae::V
    text_encoder::T
    tokenizer::K
    unet::U
    scheduler::S

    vae_scale_factor::Int
end
Flux.@functor StableDiffusion

function StableDiffusion(
    vae::V, text_encoder::T, tokenizer::K, unet::U, scheduler::S,
) where {
    V <: AutoencoderKL,
    T <: CLIPTextTransformer,
    K <: CLIPTokenizer,
    U <: UNet2DCondition,
    S <: PNDMScheduler,
}
    vae_scale_factor = 2^(length(vae.encoder.down_blocks) - 1)
    StableDiffusion(vae, text_encoder, tokenizer, unet, scheduler, vae_scale_factor)
end

# TODO n_images_per_prompt
function (sd::StableDiffusion)(
    prompt::Vector{String};
    width::Int, height::Int,
    n_inference_steps::Int = 50,
)
    prompt_embeds = _encode_prompt(sd, prompt)
    set_timesteps!(sd.scheduler, n_inference_steps)
    # TODO batch size
    latents = _prepare_latents(sd; shape=(width, height, 4, 1))
    # TODO progress bar
    for t in sd.scheduler.timesteps
        # TODO `t` must be a vector
        timestep = Int32[t]
        noise_pred = sd.unet(latents, timestep, prompt_embeds)
        latents = step!(sd.scheduler, noise_pred; t, sample=latents)
    end
    return _decode_latents(sd, latents)
end

"""
Encode prompt into text encoder hidden states.
"""
function _encode_prompt(
    sd::StableDiffusion, prompt::Vector{String};
    context_length::Int = 77,
)
    tokens, mask = tokenize(
        sd.tokenizer, prompt; add_start_end=true, context_length)
    tokens = Int32.(tokens) # TODO transfer to text encoder device

    # TODO do classifier free guidance & negative prompt
    prompt_embeds = sd.text_encoder(tokens; mask)
    # TODO n images per prompt

    prompt_embeds
end

function _prepare_latents(sd::StableDiffusion; shape::NTuple{4, Int})
    shape = (
        shape[1] ÷ sd.vae_scale_factor, shape[2] ÷ sd.vae_scale_factor,
        shape[3], shape[4])
    # TODO type, device
    latents = randn(Float32, shape)
    latents .* sd.scheduler.σ₀
end

function _decode_latents(sd::StableDiffusion, latents)
    latents .*= 1f0 / sd.vae.scaling_factor
    image = decode(sd.vae, latents)
    image = clamp!(image .* 0.5f0 .+ 0.5f0, 0f0, 1f0)
    image = permutedims(image, (3, 1, 2, 4)) # TODO transfer to host
    colorview(RGB{Float32}, image)
end

# HGF integration.

function StableDiffusion(model_name::String)
    vae = AutoencoderKL(model_name;
        state_file="vae/diffusion_pytorch_model.bin",
        config_file="vae/config.json")
    text_encoder = Diffusers.CLIPTextTransformer(model_name;
        state_file="text_encoder/pytorch_model.bin",
        config_file="text_encoder/config.json")
    tokenizer = CLIPTokenizer()
    unet = UNet2DCondition(model_name;
        state_file="unet/diffusion_pytorch_model.bin",
        config_file="unet/config.json")
    scheduler = PNDMScheduler(model_name, 4;
        config_file="scheduler/scheduler_config.json")
    StableDiffusion(vae, text_encoder, tokenizer, unet, scheduler)
end
