@testset "Load a SD ModuleList"  begin
    state_dict, cfg = load_pretrained_model("runwayml/stable-diffusion-v1-5", "unet/config.json", "unet/diffusion_pytorch_model.bin")    
    ml = torch.ModuleList([torch.Linear(rand(320, 320), rand(320, 1)), Flux.Dropout(0.1)])
    # there are no states associated with dropout, so its not present in state_dict
    torch.load_state!(ml, state_dict.down_blocks[1].attentions[1].transformer_blocks[1].attn1.to_out)
    # Forwards are not defined for ModuleList similar in PyTorch
    @test length(ml) == 2
end