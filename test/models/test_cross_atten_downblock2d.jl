@testset "Load SD CrossAttnDownBlock2D"  begin
    state_dict, cfg = load_pretrained_model("runwayml/stable-diffusion-v1-5", "unet/config.json", "unet/diffusion_pytorch_model.bin")    
    cattn = CrossAttnDownBlock2D(320, 320, 1280; num_layers=2, resnet_eps=1^-5, attn_num_head_channels=8, cross_attention_dim=768)
    models.load_state!(cattn, state_dict.down_blocks[1])
    y = cattn(ones(64, 64, 320, 1); temb=ones(1280, 1), encoder_hidden_states=ones(768, 77, 1))
    # pipe.unet.down_blocks[0](torch.ones(1, 320, 64, 64), torch.ones(1, 1280), torch.ones(1, 77, 768))
    @test y[1][1:6, 1, 1, 1] ≈ vec([3.5323777, 4.8788514, 4.8925233, 4.8956304, 4.8956304, 4.8956304]) atol=1e-3 rtol=1e-3
end

@testset "Load SD UNetMidBlock2DCrossAttn"  begin
    state_dict, cfg = load_pretrained_model("runwayml/stable-diffusion-v1-5", "unet/config.json", "unet/diffusion_pytorch_model.bin")    
    mid = UNetMidBlock2DCrossAttn(1280, 1280; resnet_eps=1^-5, attn_num_head_channels=8, cross_attention_dim=768)
    models.load_state!(mid, state_dict.mid_block)
    # pipe.unet.mid_block(torch.ones(1, 1280, 8, 8), torch.ones(1, 1280), torch.ones(1, 77, 768)).detach().numpy()[0, :6, 0, 0]
    y = mid(ones(8, 8, 1280, 1); temb=ones(1280, 1), encoder_hidden_states=ones(768, 77, 1)) 
    @test y[1, 1, 1:6, 1] ≈ vec([-2.2978039, -0.58777064, -2.1970692, -2.0825987, 3.975503, -3.1240108]) atol=1e-3 rtol=1e-3
end