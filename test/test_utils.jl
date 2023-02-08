@testset "Load SD 1.5 config and state_dict"  begin
    state_dict, cfg = load_pretrained_model("runwayml/stable-diffusion-v1-5", "unet/config.json", "unet/diffusion_pytorch_model.bin")
    @test keys(state_dict) == (:down_blocks, :mid_block, :conv_norm_out, :conv_out, :up_blocks, :conv_in, :time_embedding)
end