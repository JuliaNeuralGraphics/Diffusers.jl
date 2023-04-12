function clip_testsuite(device, fp)
    atol = fp == f32 ? 1e-3 : 1e-1
    FT = fp == f32 ? Float32 : Float16

    @testset "Embeggings" begin
        # First two values and last two values.
        x = Int32[1; 2;; 49407; 49408;;] |> device
        m = CLIP_TEXT_MODEL.embeddings |> fp |> device
        y = m(x) |> cpu
        @test size(y) == (768, 2, 2)
        @test eltype(y) == FT
        @test sum(y) ≈ 1.9741f0 atol=atol
    end

    @testset "Final layer norm" begin
        x = ones(Float32, 768, 2, 1) |> fp |> device
        m = CLIP_TEXT_MODEL.final_layer_norm |> fp |> device
        y = m(x) |> cpu
        @test eltype(y) == eltype(x)
        @test sum(y) ≈ -170.165f0 atol=atol
    end

    @testset "Encoder layers" begin
        x = ones(Float32, 768, 2, 1) |> fp |> device
        m = CLIP_TEXT_MODEL.encoder.layers[1] |> fp |> device
        y = m(x) |> cpu
        @test eltype(y) == eltype(x)
        @test sum(y) ≈ 1535f0 atol=atol
    end

    @testset "CLIP MLP" begin
        x = ones(Float32, 768, 2, 1) |> fp |> device
        m = CLIP_TEXT_MODEL.encoder.layers[1].mlp |> fp |> device
        y = m(x) |> cpu
        @test eltype(y) == eltype(x)
        @test sum(y) ≈ -5.49f0 atol=atol
    end

    @testset "Full model" begin
        x = Int32[1; 2;; 5; 6;; 49407; 49408;;]
        m = CLIP_TEXT_MODEL |> fp |> device
        y = m(x) |> cpu
        @test eltype(y) == FT
        @test sum(y) ≈ -493.05f0 atol=atol
    end
end
