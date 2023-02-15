@testset "Load SD BasicTransformerBlock & do a forward"  begin
    state_dict, cfg = load_pretrained_model("runwayml/stable-diffusion-v1-5", "unet/config.json", "unet/diffusion_pytorch_model.bin")
    tb = BasicTransformerBlock(320, 8, 40; cross_attention_dim=768)
    models.load_state!(tb, state_dict.down_blocks[1].attentions[1].transformer_blocks[1])
    y = tb(ones(320, 4096, 1), encoder_hidden_states=ones(768, 77, 1))
    # pipe.unet.down_blocks[0].attentions[0].transformer_blocks[0](torch.ones(1, 4096, 320), torch.ones(1, 77, 768))[0, 0, :5]
    @test y[1:5, 1, 1] ≈ vec([1.1293957, 0.39926898, 2.0685763, 0.07038331, 3.2459378]) atol=1e-3 rtol=1e-3
end