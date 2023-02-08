"""
    load_pretrained_model(model_name::String, config::String, bin_file::String)

Loads config and model state dict from HuggingFace library.
For example:
```julia
julia> state_dict, cfg = load_pretrained_model("runwayml/stable-diffusion-v1-5", "unet/config.json", "unet/diffusion_pytorch_model.bin")
```

"""
function load_pretrained_model(model_name::AbstractString, config::AbstractString, bin_file::AbstractString)
    cfg = Transformers.load_config_dict(model_name, config)
    state_dict = Transformers.load_state(model_name, bin_file)
    return state_dict, cfg
end
