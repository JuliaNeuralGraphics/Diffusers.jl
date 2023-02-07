# using Transformers

# clip_model_name = "openai/clip-vit-large-patch14"
# clip_config = Transformers.load_config(clip_model_name)

# clip_model_name = "runwayml/stable-diffusion-v1-5"
# @enter clip_config = Transformers.load_config("runwayml/stable-diffusion-v1-5", "unet/config.json")

# @enter clip_config = Transformers.load_config("runwayml/stable-diffusion-v1-5", "unet/config.json")

# Transformers.load_config(:bert, Transformers.load_config_dict("runwayml/stable-diffusion-v1-5", "unet/config.json"))

# using Transformers

# Transformers.load_config_dict("runwayml/stable-diffusion-v1-5", "unet/config.json")

using Diffusers: load_pretrained_model
using Revise
using Transformers


state_dict, cfg = load_pretrained_model("runwayml/stable-diffusion-v1-5", "unet/config.json", "unet/diffusion_pytorch_model.bin")

keys(state_dict.conv_in)
size(state_dict.conv_in.bias)

using Transformers

keys(state_dict)

# cfg

# state_dict = Transformers.load_state("runwayml/stable-diffusion-v1-5", "unet/diffusion_pytorch_model.bin")

# keys(state_dict)



# state_dict = Transformers.load_state("runwayml/stable-diffusion-v1-5", "vae/diffusion_pytorch_model.bin")
# Transformers.HuggingFace.hgf_model_weight("runwayml/stable-diffusion-v1-5", "vae/diffusion_pytorch_model.bin")


# Works!
# using HuggingFaceApi

# HuggingFaceApi.hf_hub_download(
#     "runwayml/stable-diffusion-v1-5", "unet/diffusion_pytorch_model.bin";
#     repo_type=nothing, revision="main",
#     auth_token="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", local_files_only=false, cache=true
# )