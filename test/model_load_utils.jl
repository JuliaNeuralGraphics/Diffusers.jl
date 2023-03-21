@testset "Load SD cross_attention & do a forward" begin
    attn = Diffusers.Attention(320; bias=false, context_dim=320, head_dim=40)
    Diffusers.load_state!(attn, STATE.down_blocks[1].attentions[1].transformer_blocks[1].attn1)

    # Manually obtained, pytorch row wise approx equals jl col wise
    # In python, pipe.unet.down_blocks[0].attentions[0].transformer_blocks[0].attn1(torch.ones(2, 4096, 320))[0, 0, :5]
    # where pipe is the StableDiffusionPipeline
    target_y = [0.15594974, 0.01136263, 0.27801704, 0.31772622, 0.63947713]
    x = ones(Float32, 320, 1, 2)
    y = attn(x, x)
    @test y[1:5, 1, 1] ≈ target_y atol=1e-5 rtol=1e-5
end

@testset "Load SD FeedForward" begin
    fwd = Diffusers.FeedForward(; dim=320)
    Diffusers.load_state!(fwd, STATE.down_blocks[1].attentions[1].transformer_blocks[1].ff)

    # pipe.unet.down_blocks[0].attentions[0].transformer_blocks[0].ff(torch.ones(1, 2, 320))[0, 0, :5].detach().numpy()
    target_y = [0.5421921, -0.00488963, 0.18569, -0.17563964, -0.0561044]
    y = fwd(ones(Float32, 320, 1, 1))
    @test y[1:5, 1, 1] ≈ target_y atol=1e-3 rtol=1e-3
end

@testset "Load SD BasicTransformerBlock & do a forward" begin
    tb = Diffusers.TransformerBlock(; dim=320, n_heads=8, head_dim=40, context_dim=768)
    Diffusers.load_state!(tb, STATE.down_blocks[1].attentions[1].transformer_blocks[1])

    # pipe.unet.down_blocks[0].attentions[0].transformer_blocks[0](torch.ones(1, 4096, 320), torch.ones(1, 77, 768))[0, 0, :5]
    target_y = [1.1293957, 0.39926898, 2.0685763, 0.07038331, 3.2459378]
    x = ones(Float32, 320, 4096, 1)
    context = ones(Float32, 768, 77, 1)
    y = tb(x, context)
    @test y[1:5, 1, 1] ≈ target_y atol=1e-3 rtol=1e-3
end

@testset "Load SD Transformer2DModel & do a forward"  begin
    tm = Diffusers.Transformer2D(;
        in_channels=320, context_dim=768, n_heads=8, head_dim=40)
    Diffusers.load_state!(tm, STATE.down_blocks[1].attentions[1])

    # pipe.unet.down_blocks[0].attentions[0](torch.ones(1, 320, 64, 64), torch.ones(1, 77, 768)).sample[0, 0, :5]
    target_y = [1.7389021, 0.795506, 1.6157904, 1.6191279, 0.6467081]
    x, context = ones(Float32, 64, 64, 320, 1), ones(Float32, 768, 77, 1)
    y = tm(x, context)
    @test y[1, 1, 1:5, 1] ≈ target_y atol=1e-3 rtol=1e-3
end

@testset "Load SD FeedForward" begin
    rs = Diffusers.ResnetBlock2D(320 => 320; time_emb_channels=1280)
    Diffusers.load_state!(rs, STATE.down_blocks[1].resnets[1])

    x, time_embedding = ones(Float32, 64, 64, 320, 1), ones(Float32, 1280, 1)

    # pipe.unet.down_blocks[0].resnets[0](torch.ones(1, 320, 64, 64), torch.ones(1, 1280)).detach().numpy()[0, :5, 0, 0]
    target_y = [1.0409687, 0.36245018, 0.92556036, 0.95282567, 1.5846546]
    y = rs(x, time_embedding)
    @test y[1, 1, 1:5, 1] ≈ target_y atol=1e-3 rtol=1e-3
end

@testset "Load SD CrossAttnDownBlock2D" begin
    cattn = Diffusers.CrossAttnDownBlock2D(; in_channels=320, out_channels=320,
    time_emb_channels=1280, n_layers=2, attn_n_heads=8, context_dim=768)
    Diffusers.load_state!(cattn, STATE.down_blocks[1])

    # pipe.unet.down_blocks[0](torch.ones(1, 320, 64, 64), torch.ones(1, 1280), torch.ones(1, 77, 768))
    x, temb, context = ones(Float32, 64, 64, 320, 1), ones(Float32, 1280, 1), ones(Float32, 768, 77, 1)
    target_y = [3.5323777, 4.8788514, 4.8925233, 4.8956304, 4.8956304, 4.8956304]
    y, rest... = cattn(x, temb, context)
    @test y[1:6, 1, 1, 1] ≈ target_y atol=1e-3 rtol=1e-3
end

@testset "Load SD CrossAttnMidBlock2D" begin
    mid = Diffusers.CrossAttnMidBlock2D(;
        in_channels=1280, time_emb_channels=1280, n_heads=8, context_dim=768)
    Diffusers.load_state!(mid, STATE.mid_block)

    # pipe.unet.mid_block(torch.ones(1, 1280, 8, 8), torch.ones(1, 1280), torch.ones(1, 77, 768)).detach().numpy()[0, :6, 0, 0]
    target_y = [-2.2978039, -0.58777064, -2.1970692, -2.0825987, 3.975503, -3.1240108]
    x, temb, context = ones(Float32, 8, 8, 1280, 1), ones(Float32, 1280, 1), ones(Float32, 768, 77, 1)
    y = mid(x, temb, context)
    @test y[1, 1, 1:6, 1] ≈ target_y atol=1e-3 rtol=1e-3
end

@testset "Load SD CrossAttnUpBlock2D" begin
    u = Diffusers.CrossAttnUpBlock2D(640=>1280, 1280, 1280; n_layers=3, attn_n_heads=8, context_dim=768)
    Diffusers.load_state!(u, STATE.up_blocks[2])
    
    # x = torch.ones(1, 1280, 16, 16)
    # pipe.unet.up_blocks[1](x, (torch.ones(1, 640, 16, 16), x, x), torch.ones(1, 1280), torch.ones(1, 77, 768))[0, :6, 0, 0]
    target_y = [-25.81815, -9.393141, -4.554784, 4.673693, 12.621728, -0.49337524]
    x = ones(Float32, 16, 16, 1280, 1)
    tl = (ones(Float32, 16, 16, 640, 1), x, x)
    y = u(x, tl, ones(Float32, 1280, 1), ones(Float32, 768, 77, 1))
    @test y[1, 1, 1:6, 1] ≈ target_y atol=1e-3 rtol=1e-3
end

@testset "Load SD UpBlock2D" begin
    u = Diffusers.UpBlock2D(1280=>1280, 1280, 1280; n_layers=3)
    Diffusers.load_state!(u, STATE.up_blocks[1])
    
    # x = torch.ones(1, 1280, 8, 8)
    # pipe.unet.up_blocks[0](x, (x, x, x), torch.ones(1, 1280)).detach().numpy()[0, :6, 0, 0]
    target_y = [0.2308563, 0.2685722, 1.0352244, 0.45586765, 1.643967, 0.10508753]
    skip = (ones(Float32, 8, 8, 1280, 1), ones(Float32, 8, 8, 1280, 1), ones(Float32, 8, 8, 1280, 1))
    x, temb = ones(Float32, 8, 8, 1280, 1), ones(Float32, 1280, 1)
    y = u(x, skip, temb)
    @test y[1, 1, 1:6, 1] ≈ target_y atol=1e-3 rtol=1e-3
end

@testset "Load SD DownBlock2D" begin
    d = Diffusers.DownBlock2D(1280=>1280, 1280; n_layers=2, add_sampler=false)
    Diffusers.load_state!(d, STATE.down_blocks[4])

    # pipe.unet.down_blocks[3](torch.ones(1, 1280, 8, 8),torch.ones(1, 1280))[0].numpy()[0, :6, 0, 0]
    target_y = [2.0826728, 1.078491, 1.1676872, 0.97314227, 0.67884475, 2.0286326]
    x, temb = ones(Float32, 8, 8, 1280, 1), ones(Float32, 1280, 1)
    y, rest... = d(x, temb)
    @test y[1, 1, 1:6, 1] ≈ target_y atol=1e-3 rtol=1e-3
end

@testset "Load a SD TimestepEmbedding with Flux & do forward" begin
    t = Diffusers.TimestepEmbedding(;in_channels=320, time_embed_dim=1280)
    Diffusers.load_state!(t, STATE.time_embedding)
    # y = pipe.unet.time_embedding(torch.ones(1, 320)).detach().numpy()[0, :6]
    x = ones(Float32, 320, 1)
    y = t(x)
    target_y = [7.0012873e-03, -6.0233027e-03, -6.9386559e-03,  5.9670270e-03, 3.6419369e-06, -4.5951810e-03]
    @test y[1:6, 1] ≈ target_y atol=1e-3 rtol=1e-3

    target_y = [0.6799572, -0.7984292, 0.57806414, -0.67470044, 0.9926904, 0.8710014]
    y = Diffusers.get_time_embedding(ones(Int, 2)*981, 320)
    @test y[1:6, 1] ≈ target_y atol=1e-3 rtol=1e-3
end

@testset "Load a SD UNet2DConditionModel with Flux & do forward" begin
    unet = Diffusers.UNet2DConditionModel(; context_dim=768)
    Diffusers.load_state!(unet, STATE)

    # y = pipe.unet(torch.ones(1, 4, 64, 64), torch.tensor(981), torch.ones(1, 77, 768)).sample.detach().numpy()[0, 0, 0, :6]
    target_y = [0.22149813, 0.16261391, 0.13246158, 0.11514825, 0.11287624, 0.11176358]
    y = unet(ones(Float32, 64, 64, 4, 1), ones(Int, 1)*981, ones(Float32, 768, 77, 1))
    @test y[1:6, 1, 1, 1] ≈ target_y atol=1e-3 rtol=1e-3
end