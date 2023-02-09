@testset "Load a SD linear layer & do forward with bias"  begin
    state_dict, cfg = load_pretrained_model("runwayml/stable-diffusion-v1-5", "unet/config.json", "unet/diffusion_pytorch_model.bin")
    ff = torch.Linear(rand(320, 320), rand(320, 1))
    torch.load_state!(ff, state_dict.down_blocks[1].attentions[1].transformer_blocks[1].attn1.to_out[1])
    y = ff(ones(320, 1, 1))
    # Harish: manually obtained, pytorch row wise approx equals jl col wise
    @test y[1:6, 1, 1] ≈ vec([2.4588253, 0.31246912, -2.4237952, 1.1872879, -0.99491394, 0.19098154]) atol=1e-5 rtol=1e-5
end

@testset "Load a SD linear layer & do forward without bias"  begin
    state_dict, cfg = load_pretrained_model("runwayml/stable-diffusion-v1-5", "unet/config.json", "unet/diffusion_pytorch_model.bin")
    ff = torch.Linear(rand(320, 320)) # no bias defined
    torch.load_state!(ff, state_dict.down_blocks[1].attentions[1].transformer_blocks[1].attn1.to_q)
    y = ff(ones(320, 1, 1))
    # Harish: manually obtained, pytorch row wise approx equals jl col wise
    @test y[1:6, 1, 1] ≈ vec([-0.0330424, 0.30586573, -0.3346526, 0.6630356, -0.39755583, 0.4008978]) atol=1e-5 rtol=1e-5
end