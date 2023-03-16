using Test
using Diffusers
using Flux

const STATE, CONFIG = Diffusers.load_pretrained_model(
    "runwayml/stable-diffusion-v1-5";
    state_file="unet/diffusion_pytorch_model.bin",
    config_file="unet/config.json")

@testset "Diffusers.jl" begin
    @testset "Layer load utils" begin
        include("layer_load_utils.jl")
    end
    @testset "Model load utils" begin
        include("model_load_utils.jl")
    end
    @testset "Schedulers" begin
        include("schedulers.jl")
    end
    @testset "Tokenizers" begin
        include("tokenizers.jl")
    end
end
