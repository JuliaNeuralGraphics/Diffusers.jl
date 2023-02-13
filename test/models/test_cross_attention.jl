@testset "Load SD cross_attention & do a forward"  begin
    state_dict, cfg = load_pretrained_model("runwayml/stable-diffusion-v1-5", "unet/config.json", "unet/diffusion_pytorch_model.bin")
    attn = CrossAttention(320; dim_head=40)
    models.load_state!(attn, state_dict.down_blocks[1].attentions[1].transformer_blocks[1].attn1)
    y = attn(ones(320, 1, 2))

    # Manually obtained, pytorch row wise approx equals jl col wise
    # In python, pipe.unet.down_blocks[0].attentions[0].transformer_blocks[0].attn1(torch.ones(2, 4096, 320))[0, 0, :5]
    # where pipe is the StableDiffusionPipeline
    @test y[1:5, 1, 1] ≈ vec([0.15594974, 0.01136263, 0.27801704, 0.31772622, 0.63947713]) atol=1e-5 rtol=1e-5
end