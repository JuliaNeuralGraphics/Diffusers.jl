const CLIP_TEXT_MODEL = CLIPTextTransformer(
    "runwayml/stable-diffusion-v1-5";
    state_file="text_encoder/pytorch_model.bin",
    config_file="text_encoder/config.json")

@testset "Embeggings" begin
    # First two values and last two values.
    y = CLIP_TEXT_MODEL.embeddings([1; 2;; 49407; 49408;;])
    @test size(y) == (2, 2)
    @test sum(y) ≈ 1.9741
end

@testset "Final layer norm" begin
    y = CLIP_TEXT_MODEL.final_layer_norm(ones(Float32, 768, 2, 1))
    @test sum(y) ≈ -170.165
end

@testset "Encoder layers" begin
    y = CLIP_TEXT_MODEL.encoder.layers[1](ones(Float32, 768, 2, 1))
    @show sum(y) ≈ 1535
end
