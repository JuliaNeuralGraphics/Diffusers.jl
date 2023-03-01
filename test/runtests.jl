using Diffusers
using Test
using Diffusers: load_pretrained_model
using Transformers
using NNlib
using Flux
using Diffusers.torch
using Diffusers.models

@testset "Diffusers.jl" begin
    @testset "torch.Conv2d" begin
        include("torch/test_conv.jl")
    end
    @testset "torch.Linear" begin
        include("torch/test_linear.jl")
    end
    @testset "torch.ModuleList" begin
        include("torch/test_modulelist.jl")
    end
    @testset "torch.LayerNorm" begin
        include("torch/test_layernorm.jl")
    end
    @testset "Flux.GroupNorm" begin
        include("torch/test_groupnorm.jl")
    end
    @testset "models.CrossAttention" begin
        include("models/test_cross_attention.jl")
    end
    @testset "models.GEGLU" begin
        include("models/test_geglu.jl")
    end
    @testset "models.FeedForward" begin
        include("models/test_feedforward.jl")
    end
    @testset "models.BasicTransformerBlock" begin
        include("models/test_basic_transformer_block.jl")
    end
    @testset "models.Transformer2DModel" begin
        include("models/test_transformer_2d.jl")
    end
    @testset "models.ResnetBlock2D" begin
        include("models/test_resnet.jl")
    end    
    @testset "utils.jl" begin
        include("test_utils.jl")
    end
end
