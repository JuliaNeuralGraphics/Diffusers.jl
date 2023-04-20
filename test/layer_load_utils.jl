@testset "Load SD unet.conv_in & do forward" begin
    conv = Conv((3, 3), 4 => 320; pad=1)
    Diffusers.load_state!(conv, STATE.conv_in)

    x = ones(Float32, 64, 64, 4, 1)
    y = conv(x)
    # Harish: manually obtained, pytorch row wise approx equals jl col wise
    # In python, pipe.unet.conv_in(torch.ones(1, 4, 64, 64))[0, 0, 0, :5]
    # where pipe is the StableDiffusionPipeline
    target_y = [-0.22148141, -0.03169854, -0.03169854, -0.03169854, -0.03169854, -0.03169854]
    @test y[1:6, 1, 1, 1] ≈ target_y atol=1e-5 rtol=1e-5
end

@testset "Load a SD linear layer & do forward with bias" begin
    m = Dense(320 => 320)
    Diffusers.load_state!(m, STATE.down_blocks[1].attentions[1].transformer_blocks[1].attn1.to_out[1])

    # Harish: manually obtained, pytorch row wise approx equals jl col wise
    target_y = [2.4588253, 0.31246912, -2.4237952, 1.1872879, -0.99491394, 0.19098154]
    y = m(ones(Float32, 320, 1))
    @test y[1:6, 1, 1] ≈ target_y atol=1e-5 rtol=1e-5
end

@testset "Load a SD linear layer & do forward without bias" begin
    m = Dense(320 => 320; bias=false)
    Diffusers.load_state!(m, STATE.down_blocks[1].attentions[1].transformer_blocks[1].attn1.to_q)

    # Harish: manually obtained, pytorch row wise approx equals jl col wise
    target_y = [-0.0330424, 0.30586573, -0.3346526, 0.6630356, -0.39755583, 0.4008978]
    y = m(ones(Float32, 320, 1))
    @test y[1:6, 1, 1] ≈ target_y atol=1e-5 rtol=1e-5
end

@testset "Load a SD ModuleList" begin
    m = Chain(Dense(320 => 320), Dropout(0.1))
    # there are no states associated with dropout, so its not present in state_dict
    Diffusers.load_state!(m, STATE.down_blocks[1].attentions[1].transformer_blocks[1].attn1.to_out)
    @test length(m) == 2
end

@testset "Load a SD layernorm & do forward with bias" begin
    ln = LayerNorm(320)
    Diffusers.load_state!(ln, STATE.down_blocks[1].attentions[1].transformer_blocks[1].norm1)

    # Harish: manually obtained, pytorch row wise approx equals jl col wise
    target_y = [0.03145087, -0.11135, 0.00409993, 0.16613194, -0.04407737]
    y = ln(ones(Float32, 320, 1, 1))
    @test y[1:5, 1, 1] ≈ target_y atol=1e-5 rtol=1e-5
end

@testset "Load a SD groupnorm with Flux & do forward" begin
    g = GroupNorm(320, 32)
    Diffusers.load_state!(g, STATE.down_blocks[1].attentions[1].norm)

    # Harish: manually obtained, pytorch row wise approx equals jl col wise
    target_y = [-0.06698608, 0.17626953, -0.22659302, 0.03451538, -0.01315308]
    y = g(ones(Float32, 3, 3, 320, 1))
    @test y[1, 1, 1:5, 1] ≈ target_y atol=1e-3 rtol=1e-3
end
