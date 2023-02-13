@testset "Load SD FeedForward"  begin
    state_dict, cfg = load_pretrained_model("runwayml/stable-diffusion-v1-5", "unet/config.json", "unet/diffusion_pytorch_model.bin")    
    ff = FeedForward(320; dropout=0.0f0, activation_fn="geglu", final_dropout=false)
    models.load_state!(ff, state_dict.down_blocks[1].attentions[1].transformer_blocks[1].ff)
    y = ff(ones(320, 1, 1))
    # pipe.unet.down_blocks[0].attentions[0].transformer_blocks[0].ff(torch.ones(1, 2, 320))[0, 0, :5].detach().numpy()
    @test y[1:5, 1, 1] ≈ vec([0.5421921, -0.00488963, 0.18569, -0.17563964, -0.0561044]) atol=1e-3 rtol=1e-3
end