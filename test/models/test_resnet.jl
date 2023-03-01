@testset "Load SD FeedForward"  begin
    state_dict, cfg = load_pretrained_model("runwayml/stable-diffusion-v1-5", "unet/config.json", "unet/diffusion_pytorch_model.bin")    
    rs = ResnetBlock2D(320; out_channels=320, temb_channels=1280, groups_out=32, )
    models.load_state!(rs, state_dict.down_blocks[1].resnets[1])
    y = rs(ones(64, 64, 320, 1), ones(1280, 1))
    # pipe.unet.down_blocks[0].resnets[0](torch.ones(1, 320, 64, 64), torch.ones(1, 1280)).detach().numpy()[0, :5, 0, 0]
    @test y[1, 1, 1:5, 1] ≈ vec([1.0409687, 0.36245018, 0.92556036, 0.95282567, 1.5846546]) atol=1e-3 rtol=1e-3
end