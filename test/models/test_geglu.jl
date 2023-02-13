@testset "Load SD GEGLU & do a forward"  begin
    state_dict, cfg = load_pretrained_model("runwayml/stable-diffusion-v1-5", "unet/config.json", "unet/diffusion_pytorch_model.bin")
    geglu = GEGLU(320, 1280)
    models.load_state!(geglu, state_dict.down_blocks[1].attentions[1].transformer_blocks[1].ff.net[1])
    y = geglu(ones(320, 2, 1))

    # Manually obtained, pytorch row wise approx equals jl col wise
    # In python, pipe.unet.down_blocks[0].attentions[0].transformer_blocks[0].ff.net[0](torch.ones(1, 2, 320))[0, 0, :5]
    # where pipe is the StableDiffusionPipeline
    @test y[1:5, 1, 1] ≈ vec([0.02284688, -0.01552063, 0.05385243, 0.12253999, -0.02233481]) atol=1e-3 rtol=1e-3
end