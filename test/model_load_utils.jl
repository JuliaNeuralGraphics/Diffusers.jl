function model_load_testsuite(device, fp)
    atol = fp == f32 ? 1e-3 : 1e-1

    @testset "Attention" begin
        attn = Diffusers.Attention(320; bias=false, context_dim=320, head_dim=40)
        Diffusers.load_state!(attn, STATE.down_blocks[1].attentions[1].transformer_blocks[1].attn1)
        m = attn |> fp |> device

        # pipe.unet.down_blocks[0].attentions[0].transformer_blocks[0].attn1(torch.ones(2, 4096, 320))[0, 0, :5]
        target_y = [0.15594974, 0.01136263, 0.27801704, 0.31772622, 0.63947713] |> fp
        x = ones(Float32, 320, 1, 2) |> fp |> device
        y = m(x, x) |> cpu

        @test eltype(x) == eltype(y)
        @test !any(isnan.(y))
        @test y[1:5, 1, 1] ≈ target_y atol=atol
    end

    @testset "FeedForward" begin
        fwd = Diffusers.FeedForward(; dim=320)
        Diffusers.load_state!(fwd, STATE.down_blocks[1].attentions[1].transformer_blocks[1].ff)
        m = fwd |> fp |> device

        # pipe.unet.down_blocks[0].attentions[0].transformer_blocks[0].ff(torch.ones(1, 2, 320))[0, 0, :5].detach().numpy()
        target_y = [0.5421921, -0.00488963, 0.18569, -0.17563964, -0.0561044] |> fp
        x = ones(Float32, 320, 1, 1) |> fp |> device
        y = m(x) |> cpu

        @test eltype(x) == eltype(y)
        @test !any(isnan.(y))
        @test cpu(y)[1:5, 1, 1] ≈ target_y atol=atol
    end

    @testset "TransformerBlock" begin
        tb = Diffusers.TransformerBlock(; dim=320, n_heads=8, head_dim=40, context_dim=768)
        Diffusers.load_state!(tb, STATE.down_blocks[1].attentions[1].transformer_blocks[1])
        m = tb |> fp |> device

        # pipe.unet.down_blocks[0].attentions[0].transformer_blocks[0](torch.ones(1, 4096, 320), torch.ones(1, 77, 768))[0, 0, :5]
        target_y = [1.1293957, 0.39926898, 2.0685763, 0.07038331, 3.2459378] |> fp
        x = ones(Float32, 320, 4096, 1) |> fp |> device
        context = ones(Float32, 768, 77, 1) |> fp |> device
        y = m(x, context) |> cpu

        @test eltype(x) == eltype(y)
        @test !any(isnan.(y))
        @test y[1:5, 1, 1] ≈ target_y atol=atol
    end

    @testset "Load SD Transformer2DModel & do a forward"  begin
        tm = Diffusers.Transformer2D(; in_channels=320, context_dim=768, n_heads=8, head_dim=40)
        Diffusers.load_state!(tm, STATE.down_blocks[1].attentions[1])
        m = tm |> fp |> device

        # pipe.unet.down_blocks[0].attentions[0](torch.ones(1, 320, 64, 64), torch.ones(1, 77, 768)).sample[0, 0, :5]
        target_y = [1.7389021, 0.795506, 1.6157904, 1.6191279, 0.6467081] |> fp
        x = ones(Float32, 64, 64, 320, 1) |> fp |> device
        context = ones(Float32, 768, 77, 1) |> fp |> device
        y = m(x, context) |> cpu

        @test eltype(x) == eltype(y)
        @test !any(isnan.(y))
        @test y[1, 1, 1:5, 1] ≈ target_y atol=atol
    end

    @testset "Load SD ResnetBlock2D" begin
        rs = Diffusers.ResnetBlock2D(320 => 320; time_emb_channels=1280)
        Diffusers.load_state!(rs, STATE.down_blocks[1].resnets[1])
        m = rs |> fp |> device

        x = ones(Float32, 64, 64, 320, 1) |> fp |> device
        time_embedding = ones(Float32, 1280, 1) |> fp |> device

        # pipe.unet.down_blocks[0].resnets[0](torch.ones(1, 320, 64, 64), torch.ones(1, 1280)).detach().numpy()[0, :5, 0, 0]
        target_y = [1.0409687, 0.36245018, 0.92556036, 0.95282567, 1.5846546] |> fp
        y = m(x, time_embedding) |> cpu

        @test eltype(x) == eltype(y)
        @test !any(isnan.(y))
        @test y[1, 1, 1:5, 1] ≈ target_y atol=atol
    end

    @testset "Load SD CrossAttnDownBlock2D" begin
        cattn = Diffusers.CrossAttnDownBlock2D(320 => 320;
            time_emb_channels=1280, n_layers=2, n_heads=8, context_dim=768)
        Diffusers.load_state!(cattn, STATE.down_blocks[1])
        m = cattn |> fp |> device

        x = ones(Float32, 64, 64, 320, 1) |> fp |> device
        temb = ones(Float32, 1280, 1) |> fp |> device
        context = ones(Float32, 768, 77, 1) |> fp |> device
        # pipe.unet.down_blocks[0](torch.ones(1, 320, 64, 64), torch.ones(1, 1280), torch.ones(1, 77, 768))
        target_y = [3.5323777, 4.8788514, 4.8925233, 4.8956304, 4.8956304, 4.8956304] |> fp
        y = m(x, temb, context)[1] |> cpu

        @test eltype(x) == eltype(y)
        @test !any(isnan.(y))
        @test y[1:6, 1, 1, 1] ≈ target_y atol=atol
    end

    @testset "Load SD CrossAttnMidBlock2D" begin
        mid = Diffusers.CrossAttnMidBlock2D(;
            in_channels=1280, time_emb_channels=1280, n_heads=8, context_dim=768)
        Diffusers.load_state!(mid, STATE.mid_block)
        m = mid |> fp |> device

        # pipe.unet.mid_block(torch.ones(1, 1280, 8, 8), torch.ones(1, 1280), torch.ones(1, 77, 768)).detach().numpy()[0, :6, 0, 0]
        target_y = [-2.2978039, -0.58777064, -2.1970692, -2.0825987, 3.975503, -3.1240108] |> fp
        x = ones(Float32, 8, 8, 1280, 1) |> fp |> device
        temb = ones(Float32, 1280, 1) |> fp |> device
        context = ones(Float32, 768, 77, 1) |> fp |> device
        y = m(x, temb, context) |> cpu

        @test eltype(x) == eltype(y)
        @test !any(isnan.(y))
        @test y[1, 1, 1:6, 1] ≈ target_y atol=atol
    end

    @testset "Load SD CrossAttnUpBlock2D" begin
        u = Diffusers.CrossAttnUpBlock2D(640=>1280, 1280, 1280; n_layers=3, attn_n_heads=8, context_dim=768)
        Diffusers.load_state!(u, STATE.up_blocks[2])
        m = u |> fp |> device

        # x = torch.ones(1, 1280, 16, 16)
        # pipe.unet.up_blocks[1](x, (torch.ones(1, 640, 16, 16), x, x), torch.ones(1, 1280), torch.ones(1, 77, 768))[0, :6, 0, 0]
        target_y = [-25.81815, -9.393141, -4.554784, 4.673693, 12.621728, -0.49337524] |> fp
        x = ones(Float32, 16, 16, 1280, 1) |> fp |> device
        skips = (x, x, ones(Float32, 16, 16, 640, 1)) |> fp |> device
        temb = ones(Float32, 1280, 1) |> fp |> device
        context = ones(Float32, 768, 77, 1) |> fp |> device
        y = m(x, skips, temb, context)[1] |> cpu

        @test eltype(x) == eltype(y)
        @test !any(isnan.(y))
        @test y[1, 1, 1:6, 1] ≈ target_y atol=atol
    end

    @testset "Load SD UpBlock2D" begin
        u = Diffusers.UpBlock2D(1280=>1280, 1280, 1280; n_layers=3)
        Diffusers.load_state!(u, STATE.up_blocks[1])
        m = u |> fp |> device

        # pipe.unet.up_blocks[0](x, (x, x, x), torch.ones(1, 1280)).detach().numpy()[0, :6, 0, 0]
        target_y = [0.2308563, 0.2685722, 1.0352244, 0.45586765, 1.643967, 0.10508753] |> fp
        skip = (ones(Float32, 8, 8, 1280, 1), ones(Float32, 8, 8, 1280, 1), ones(Float32, 8, 8, 1280, 1)) |> fp |> device
        x = ones(Float32, 8, 8, 1280, 1) |> fp |> device
        temb = ones(Float32, 1280, 1) |> fp |> device
        y = m(x, skip, temb)[1] |> cpu

        @test eltype(x) == eltype(y)
        @test !any(isnan.(y))
        @test y[1, 1, 1:6, 1] ≈ target_y atol=atol
    end

    @testset "Load SD DownBlock2D" begin
        d = Diffusers.DownBlock2D(1280 => 1280, 1280; n_layers=2, add_downsample=false)
        Diffusers.load_state!(d, STATE.down_blocks[4])
        m = d |> fp |> device

        # pipe.unet.down_blocks[3](torch.ones(1, 1280, 8, 8),torch.ones(1, 1280))[0].numpy()[0, :6, 0, 0]
        target_y = [2.0826728, 1.078491, 1.1676872, 0.97314227, 0.67884475, 2.0286326] |> fp
        x = ones(Float32, 8, 8, 1280, 1) |> fp |> device
        temb = ones(Float32, 1280, 1) |> fp |> device
        y = m(x, temb)[1] |> cpu

        @test eltype(x) == eltype(y)
        @test !any(isnan.(y))
        @test y[1, 1, 1:6, 1] ≈ target_y atol=atol
    end

    @testset "Load a SD UNet2DCondition with Flux & do forward" begin
        unet = Diffusers.UNet2DCondition(; context_dim=768)
        Diffusers.load_state!(unet, STATE)
        m = unet |> fp |> device

        # y = pipe.unet(torch.ones(1, 4, 64, 64), torch.tensor(981), torch.ones(1, 77, 768)).sample.detach().numpy()[0, 0, 0, :6]
        target_y = [0.22149813, 0.16261391, 0.13246158, 0.11514825, 0.11287624, 0.11176358] |> fp
        x = ones(Float32, 64, 64, 4, 1) |> fp |> device
        timesteps = (ones(Int32, 1) * Int32(981)) |> device
        text_embedding = ones(Float32, 768, 77, 1) |> fp |> device

        y = m(x, timesteps, text_embedding) |> cpu

        @test eltype(x) == eltype(y)
        @test !any(isnan.(y))
        @test y[1:6, 1, 1, 1] ≈ target_y atol=atol
    end
end
