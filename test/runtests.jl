using Test
using Diffusers
using Flux

const STATE_DICT, CFG = load_pretrained_model(
    "runwayml/stable-diffusion-v1-5",
    "unet/config.json",
    "unet/diffusion_pytorch_model.bin")

@testset "Diffusers.jl" begin
    # @testset "Layer load utils" begin
    #     include("layer_load_utils.jl")
    # end

    @testset "Model load utils" begin
        include("model_load_utils.jl")
    end

    # @testset "models.GEGLU" begin
    #     include("models/test_geglu.jl")
    # end
    # @testset "models.FeedForward" begin
    #     include("models/test_feedforward.jl")
    # end
    # @testset "models.BasicTransformerBlock" begin
    #     include("models/test_basic_transformer_block.jl")
    # end
    # @testset "models.Transformer2DModel" begin
    #     include("models/test_transformer_2d.jl")
    # end
    # @testset "models.ResnetBlock2D" begin
    #     include("models/test_resnet.jl")
    # end
    # @testset "utils.jl" begin
    #     include("test_utils.jl")
    # end
end
