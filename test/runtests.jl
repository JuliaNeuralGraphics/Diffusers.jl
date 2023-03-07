using Test
using Diffusers
using Flux

const STATE_DICT, CFG = Diffusers.load_pretrained_model(
    "runwayml/stable-diffusion-v1-5",
    "unet/config.json",
    "unet/diffusion_pytorch_model.bin")

@testset "Diffusers.jl" begin
    @testset "Layer load utils" begin
        include("layer_load_utils.jl")
    end
    @testset "Model load utils" begin
        include("model_load_utils.jl")
    end
end
