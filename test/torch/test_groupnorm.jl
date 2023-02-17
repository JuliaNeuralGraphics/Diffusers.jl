@testset "Load a SD groupnorm with Flux & do forward"  begin
    state_dict, cfg = load_pretrained_model("runwayml/stable-diffusion-v1-5", "unet/config.json", "unet/diffusion_pytorch_model.bin")
    g = Flux.GroupNorm(320, 32)
    torch.load_state!(g, state_dict.down_blocks[1].attentions[1].norm)
    y = g(ones(3, 3, 320, 1))
    # Harish: manually obtained, pytorch row wise approx equals jl col wise
    @test y[1, 1, 1:5, 1] ≈ vec([-0.06698608, 0.17626953, -0.22659302, 0.03451538, -0.01315308]) atol=1e-3 rtol=1e-3
end