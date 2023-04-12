function scheduler_testsuite(device, fp)
    FP = fp == f32 ? Float32 : Float16
    atol = fp == f32 ? 1e-3 : 1e-1

    @testset "PNDM Scheduler" begin
        pndm = Diffusers.PNDMScheduler(4; n_train_timesteps=50) |> fp |> device
        Diffusers.set_timesteps!(pndm, 15)

        @test eltype(pndm.α̂) == FP
        @test eltype(pndm.x_current) == FP

        @test length(pndm.timesteps) == 24
        @test pndm.prk_timesteps == [42, 40, 40, 39, 39, 37, 37, 36, 36, 34, 34, 33]
        @test pndm.plms_timesteps == [33, 30, 27, 24, 21, 18, 15, 12, 9, 6, 3, 0]
        @test pndm.timesteps == [
            42, 40, 40, 39, 39, 37, 37, 36, 36, 34, 34, 33,
            33, 30, 27, 24, 21, 18, 15, 12, 9, 6, 3, 0]

        x = ones(Float32, 2, 2, 3, 1) |> fp |> device
        sample = ones(Float32, 2, 2, 3, 1) |> fp |> device

        # Test PRK.
        prev_sample = Diffusers.step!(pndm, x; t=0, sample)
        @test eltype(prev_sample) == FP
        @test sum(prev_sample) ≈ 6.510407f0 atol=atol

        for t in 1:11
            Diffusers.step!(pndm, x; t, sample)
        end

        # Test PLMS.
        prev_sample = Diffusers.step!(pndm, x; t=12, sample)
        @test eltype(prev_sample) == FP
        @test sum(prev_sample) ≈ 11.563744f0 atol=atol

        ξ = ones(Float32, 2, 2, 3, 1) |> fp |> device
        timesteps = [1]
        y = Diffusers.add_noise(pndm, x, ξ, timesteps)
        @test eltype(y) == FP
        @test size(y) == size(x)

        timesteps = [1, 2, 3, 4]
        y = Diffusers.add_noise(pndm, x, ξ, timesteps)
        @test eltype(y) == FP
        @test size(y, 4) == length(timesteps)
        @test size(y)[1:3] == size(x)[1:3]
    end

    @testset "Load from HGF" begin
        Diffusers.PNDMScheduler("runwayml/stable-diffusion-v1-5", 4;
            config_file="scheduler/scheduler_config.json")
    end
end
