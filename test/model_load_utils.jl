@testset "Load SD cross_attention & do a forward" begin
    attn = Diffusers.CrossAttention(; dim=320, head_dim=40)
    Diffusers.load_state!(attn, STATE_DICT.down_blocks[1].attentions[1].transformer_blocks[1].attn1)

    # Manually obtained, pytorch row wise approx equals jl col wise
    # In python, pipe.unet.down_blocks[0].attentions[0].transformer_blocks[0].attn1(torch.ones(2, 4096, 320))[0, 0, :5]
    # where pipe is the StableDiffusionPipeline
    target_y = [0.15594974, 0.01136263, 0.27801704, 0.31772622, 0.63947713]
    x = ones(Float32, 320, 1, 2)
    y = attn(x, x)
    @test y[1:5, 1, 1] ≈ target_y atol=1e-5 rtol=1e-5
end
