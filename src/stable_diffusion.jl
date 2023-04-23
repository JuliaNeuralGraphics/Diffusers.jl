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

function get_backend(sd::StableDiffusion)
    typeof(sd.unet.sin_embedding.emb) <: Array ? cpu : gpu
end

Base.eltype(sd::StableDiffusion) = eltype(sd.unet.sin_embedding.emb)

function (sd::StableDiffusion)(
    prompt::Vector{String}, negative_prompt::Vector{String} = String[];
    n_inference_steps::Int = 20,
    n_images_per_prompt::Int = 1,
    guidance_scale::Float32 = 7.5f0,
)
    width, height = 512, 512

    classifier_free_guidance = guidance_scale > 1f0
    prompt_embeds = _encode_prompt(
        sd, prompt, negative_prompt; n_images_per_prompt,
        classifier_free_guidance)
    GC.gc()

    set_timesteps!(sd.scheduler, n_inference_steps)
    GC.gc()

    batch = length(prompt) * n_images_per_prompt
    latents = _prepare_latents(sd; shape=(width, height, 4, batch))
    GC.gc()

    bar = get_pb(length(sd.scheduler.timesteps), "Diffusion process:")
    for t in sd.scheduler.timesteps
        timestep = Int32[t] |> get_backend(sd)
        # Double latents for classifier free guidance.
        latent_inputs = classifier_free_guidance ? cat(latents, latents; dims=4) : latents
        noise_pred = sd.unet(latent_inputs, timestep, prompt_embeds)
        GC.gc()

        # Perform guidance.
        if classifier_free_guidance
            noise_pred_uncond, noise_pred_text = MLUtils.chunk(noise_pred, 2; dims=4)
            noise_pred = noise_pred_uncond .+ eltype(sd)(guidance_scale) .* (noise_pred_text .- noise_pred_uncond)
        end

        latents = step!(sd.scheduler, noise_pred; t, sample=latents)
        GC.gc()
        next!(bar)
    end
    return _decode_latents(sd, latents)
end

"""
Encode prompt into text encoder hidden states.
"""
function _encode_prompt(
    sd::StableDiffusion, prompt::Vector{String},
    negative_prompt::Vector{String};
    context_length::Int = 77,
    n_images_per_prompt::Int,
    classifier_free_guidance::Bool,
)
    tokens, mask = tokenize(
        sd.tokenizer, prompt; add_start_end=true, context_length)
    tokens = Int32.(tokens) |> get_backend(sd)

    prompt_embeds = sd.text_encoder(tokens #=, mask =#) # TODO conditionally use mask
    _, seq_len, batch = size(prompt_embeds)
    prompt_embeds = repeat(prompt_embeds; outer=(1, n_images_per_prompt, 1))
    prompt_embeds = reshape(prompt_embeds, :, seq_len, batch * n_images_per_prompt)

    if classifier_free_guidance
        negative_prompt = isempty(negative_prompt) ?
            fill("", length(prompt)) : negative_prompt
        @assert length(negative_prompt) == length(prompt)

        tokens, mask = tokenize(
            sd.tokenizer, negative_prompt; add_start_end=true, context_length)
        tokens = Int32.(tokens) |> get_backend(sd)

        negative_prompt_embeds = sd.text_encoder(tokens #=, mask =#) # TODO conditionally use mask
        _, seq_len, batch = size(negative_prompt_embeds)
        negative_prompt_embeds = repeat(negative_prompt_embeds; outer=(1, n_images_per_prompt, 1))
        negative_prompt_embeds = reshape(negative_prompt_embeds, :, seq_len, batch * n_images_per_prompt)

        # For classifier free guidance we need to do 2 forward passes.
        # Instead, concatenate embeds together and do 1.
        prompt_embeds = cat(negative_prompt_embeds, prompt_embeds; dims=3)
    end
    prompt_embeds
end

function _prepare_latents(sd::StableDiffusion; shape::NTuple{4, Int})
    shape = (
        shape[1] ÷ sd.vae_scale_factor,
        shape[2] ÷ sd.vae_scale_factor, shape[3], shape[4])
    FP = eltype(sd)
    latents = randn(FP, shape) |> get_backend(sd)
    isone(sd.scheduler.σ₀) || return latents
    latents .* FP(sd.scheduler.σ₀)
end

function _decode_latents(sd::StableDiffusion, latents)
    FP = eltype(sd)
    latents .*= FP(1f0 / sd.vae.scaling_factor)
    image = decode(sd.vae, latents)
    host_image = image |> cpu
    host_image = clamp!(Float32.(host_image) .* 0.5f0 .+ 0.5f0, 0f0, 1f0)
    host_image = permutedims(host_image, (3, 1, 2, 4))
    colorview(RGB{Float32}, host_image)
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

# Truncate type to improve stacktrace readability.
# TODO there should be more generic way.
function Base.show(io::IO, ::Type{<: StableDiffusion{V, T, K, U, S}}) where {
    V, T, K, U, S,
}
    print(io, "StableDiffusion{$(V.name.wrapper){…}, $(T.name.wrapper){…}, $(K.name.wrapper){…}, $(U.name.wrapper){…}, $(S.name.wrapper){…}}")
end
