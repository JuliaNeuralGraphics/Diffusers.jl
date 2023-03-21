const CLIP_TEXT_MODEL = Diffusers.CLIPTextTransformer(
    "runwayml/stable-diffusion-v1-5";
    state_file="text_encoder/pytorch_model.bin",
    config_file="text_encoder/config.json")

@testset "Embeggings" begin
    # First two values and last two values.
    y = CLIP_TEXT_MODEL.embeddings(Int32[1; 2;; 49407; 49408;;])
    @test size(y) == (768, 2, 2)
    @test sum(y) ≈ 1.9741f0
end

@testset "Final layer norm" begin
    y = CLIP_TEXT_MODEL.final_layer_norm(ones(Float32, 768, 2, 1))
    @test sum(y) ≈ -170.165f0
end

@testset "Encoder layers" begin
    y = CLIP_TEXT_MODEL.encoder.layers[1](ones(Float32, 768, 2, 1))
    @test sum(y) ≈ 1535f0
end

@testset "CLIP MLP" begin
    x = ones(Float32, 768, 2, 1)
    @test sum(CLIP_TEXT_MODEL.encoder.layers[1].mlp(x)) ≈ -5.49f0
end

@testset "Full model" begin
    x = Int32[1; 2;; 5; 6;; 49407; 49408;;]
    y = CLIP_TEXT_MODEL(x)
    @test sum(y) ≈ -493.05f0
end
