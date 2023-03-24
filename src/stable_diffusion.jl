struct StableDiffusion{V, T, K, U, S}
    vae::V
    text_encoder::T
    tokenizer::K
    unet::U
    scheduler::S

    vae_scale_factor::Float32
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
    vae_scale_factor = 2f0^(length(vae.encoder.down_blocks) - 1)
    StableDiffusion(vae, text_encoder, tokenizer, unet, scheduler, vae_scale_factor)
end

# TODO n_images_per_prompt
function (sd::StableDiffusion)(
    prompt::Vector{String};
    width::Int, height::Int,
    n_inference_steps::Int = 50,
)
    prompt_embeds = _encode_prompt(sd, prompt; n_images_per_prompt)
    set_timesteps!(sd.scheduler, n_inference_steps)

    # TODO prepare latents: https://github.com/huggingface/diffusers/blob/c7da8fd23359a22d0df2741688b5b4f33c26df21/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L642
    # TODO denoising loop
    # TODO decode latents
    # TODO save image
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
    @show size(prompt_embeds)
    # TODO n images per prompt

    prompt_embeds
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
