@testset "Load a SD layernorm & do forward with bias"  begin
    state_dict, cfg = load_pretrained_model("runwayml/stable-diffusion-v1-5", "unet/config.json", "unet/diffusion_pytorch_model.bin")
    ln = torch.LayerNorm(320)
    torch.load_state!(ln, state_dict.down_blocks[1].attentions[1].transformer_blocks[1].norm1)
    y = ln(ones(320, 1, 1))
    # Harish: manually obtained, pytorch row wise approx equals jl col wise
    @test y[1:5, 1, 1] ≈ vec([0.03145087, -0.11135, 0.00409993, 0.16613194, -0.04407737]) atol=1e-5 rtol=1e-5
end