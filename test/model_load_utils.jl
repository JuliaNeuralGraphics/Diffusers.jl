# @testset "Load SD cross_attention & do a forward" begin
#     attn = Diffusers.CrossAttention(; dim=320, head_dim=40)
#     Diffusers.load_state!(attn, STATE_DICT.down_blocks[1].attentions[1].transformer_blocks[1].attn1)

#     # Manually obtained, pytorch row wise approx equals jl col wise
#     # In python, pipe.unet.down_blocks[0].attentions[0].transformer_blocks[0].attn1(torch.ones(2, 4096, 320))[0, 0, :5]
#     # where pipe is the StableDiffusionPipeline
#     target_y = [0.15594974, 0.01136263, 0.27801704, 0.31772622, 0.63947713]
#     x = ones(Float32, 320, 1, 2)
#     y = attn(x)
#     @test y[1:5, 1, 1] ≈ target_y atol=1e-5 rtol=1e-5
# end

# @testset "Load SD FeedForward" begin
#     fwd = Diffusers.FeedForward(; dim=320)
#     Diffusers.load_state!(fwd, STATE_DICT.down_blocks[1].attentions[1].transformer_blocks[1].ff)

#     # pipe.unet.down_blocks[0].attentions[0].transformer_blocks[0].ff(torch.ones(1, 2, 320))[0, 0, :5].detach().numpy()
#     target_y = [0.5421921, -0.00488963, 0.18569, -0.17563964, -0.0561044]
#     y = fwd(ones(Float32, 320, 1, 1))
#     @test y[1:5, 1, 1] ≈ target_y atol=1e-3 rtol=1e-3
# end

# @testset "Load SD BasicTransformerBlock & do a forward" begin
#     tb = Diffusers.TransformerBlock(; dim=320, n_heads=8, head_dim=40, context_dim=768)
#     Diffusers.load_state!(tb, STATE_DICT.down_blocks[1].attentions[1].transformer_blocks[1])

#     # pipe.unet.down_blocks[0].attentions[0].transformer_blocks[0](torch.ones(1, 4096, 320), torch.ones(1, 77, 768))[0, 0, :5]
#     target_y = [1.1293957, 0.39926898, 2.0685763, 0.07038331, 3.2459378]
#     x = ones(Float32, 320, 4096, 1)
#     context = ones(Float32, 768, 77, 1)
#     y = tb(x, context)
#     @test y[1:5, 1, 1] ≈ target_y atol=1e-3 rtol=1e-3
# end

@testset "Load SD Transformer2DModel & do a forward"  begin
    tm = Transformer2DModel(;
        num_attention_heads=8, attention_head_dim=40, in_channels=320, cross_attention_dim=768,)
    Diffusers.load_state!(tm, STATE_DICT.down_blocks[1].attentions[1])

    # pipe.unet.down_blocks[0].attentions[0](torch.ones(1, 320, 64, 64), torch.ones(1, 77, 768)).sample[0, 0, :5]
    target_y = [1.7389021, 0.795506, 1.6157904, 1.6191279, 0.6467081]
    y = tm(ones(64, 64, 320, 1); encoder_hidden_states=ones(768, 77, 1))
    @test y.sample[1, 1, 1:5, 1] ≈ target_y atol=1e-3 rtol=1e-3
end
