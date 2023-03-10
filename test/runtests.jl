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

    @testset "PNDM Scheduler" begin
        pndm = Diffusers.PNDMScheduler(4; n_train_steps=50)
        Diffusers.set_timesteps!(pndm, 15)

        @test length(pndm.timesteps) == 24
        @test pndm.prk_timesteps == [42, 40, 40, 39, 39, 37, 37, 36, 36, 34, 34, 33]
        @test pndm.plms_timesteps == [33, 30, 27, 24, 21, 18, 15, 12, 9, 6, 3, 0]
        @test pndm.timesteps == [
            42, 40, 40, 39, 39, 37, 37, 36, 36, 34, 34, 33,
            33, 30, 27, 24, 21, 18, 15, 12, 9, 6, 3, 0]

        x = ones(Float32, 2, 2, 3, 1)
        sample = ones(Float32, 2, 2, 3, 1)

        # Test PRK.
        prev_sample = Diffusers.step!(pndm, x; t=0, sample)
        @test sum(prev_sample) ≈ 6.510407f0

        for t in 1:11
            Diffusers.step!(pndm, x; t, sample)
        end

        # Test PLMS.
        ns = Diffusers.step!(pndm, x; t=12, sample)
        @test sum(ns) ≈ 11.563744f0

        ξ = ones(Float32, 2, 2, 3, 1)
        timesteps = [1]
        y = Diffusers.add_noise(pndm, x, ξ, timesteps)
        @test size(y) == size(x)

        timesteps = [1, 2, 3, 4]
        y = Diffusers.add_noise(pndm, x, ξ, timesteps)
        @test size(y, 4) == length(timesteps)
        @test size(y)[1:3] == size(x)[1:3]
    end
end
