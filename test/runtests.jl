using Test
using Diffusers
using Flux

const STATE, CONFIG = Diffusers.load_pretrained_model(
    "runwayml/stable-diffusion-v1-5";
    state_file="unet/diffusion_pytorch_model.bin",
    config_file="unet/config.json")

const CLIP_TEXT_MODEL = Diffusers.CLIPTextTransformer(
    "runwayml/stable-diffusion-v1-5";
    state_file="text_encoder/pytorch_model.bin",
    config_file="text_encoder/config.json")

include("model_load_utils.jl")
include("clip.jl")
include("schedulers.jl")

@testset verbose=true "Diffusers.jl" begin
    for fp in (f32, f16), device in (cpu,)
        @testset verbose=true "Device: $device, Precision: $fp" begin
            @testset "Model layers" begin
                model_load_testsuite(device, fp)
            end
            @testset "CLIP model" begin
                clip_testsuite(device, fp)
            end
            @testset "Schedulers" begin
                scheduler_testsuite(device, fp)
            end
        end
    end
    @testset "Tokenizers" begin
        include("tokenizers.jl")
    end
    # @testset "Layer load utils" begin
    #     include("layer_load_utils.jl")
    # end
end
