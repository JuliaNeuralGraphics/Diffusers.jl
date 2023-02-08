using Diffusers
using Test
using Diffusers: load_pretrained_model
using Transformers
using NNlib
using Diffusers.torch

@testset "Diffusers.jl" begin
    @testset "torch.Conv2d" begin
        include("torch/test_conv.jl")
    end
    @testset "utils.jl" begin
        include("test_utils.jl")
    end    
end
