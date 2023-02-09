using Diffusers
using Test
using Diffusers: load_pretrained_model
using Transformers
using NNlib
using Flux
using Diffusers.torch

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
    @testset "models.CrossAttention" begin
        include("models/test_cross_attention.jl")
    end    
    @testset "utils.jl" begin
        include("test_utils.jl")
    end    
end
