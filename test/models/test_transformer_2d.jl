@testset "Load SD Transformer2DModel & do a forward"  begin
    state_dict, cfg = load_pretrained_model("runwayml/stable-diffusion-v1-5", "unet/config.json", "unet/diffusion_pytorch_model.bin")
    tm = Transformer2DModel(; num_attention_heads=8, attention_head_dim=40, in_channels=320, cross_attention_dim=768,)
    models.load_state!(tm, state_dict.down_blocks[1].attentions[1])
    y = tm(ones(64, 64, 320, 1); encoder_hidden_states=ones(768, 77, 1))
    # pipe.unet.down_blocks[0].attentions[0](torch.ones(1, 320, 64, 64), torch.ones(1, 77, 768)).sample[0, 0, :5]
    @test y.sample[1, 1, 1:5, 1] ≈ vec([1.7389021, 0.795506,  1.6157904, 1.6191279, 0.6467081]) atol=1e-3 rtol=1e-3
end