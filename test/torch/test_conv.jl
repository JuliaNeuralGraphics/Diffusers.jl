
@testset "Load SD unet.conv_in & do forward"  begin
    state_dict, cfg = load_pretrained_model("runwayml/stable-diffusion-v1-5", "unet/config.json", "unet/diffusion_pytorch_model.bin")
    conv = torch.Conv2d(rand(3, 3, 4, 320), rand(1, 1, 320, 1); padding=(1,1), stride=1, dilation=1, flipkernel=true)
    # Load conv
    torch.load_state!(conv, state_dict.conv_in)
    # Forward 
    x = ones(64, 64, 4, 1)
    # Pass cdims manually
    cdims = DenseConvDims(x, conv.weight, padding=(1,1), stride=1, dilation=1, flipkernel=true)
    y = conv(x; cdims=cdims)
    # Forward creates cdims 
    z = conv(x)
    @test z == y
    # Harish: manually obtained, pytorch row wise approx equals jl col wise
    # In python, pipe.unet.conv_in(torch.ones(1, 4, 64, 64))[0, 0, 0, :5]
    # where pipe is the StableDiffusionPipeline
    @test y[1:6, 1, 1, 1] ≈ vec([-0.22148141, -0.03169854, -0.03169854, -0.03169854, -0.03169854, -0.03169854]) atol=1e-5 rtol=1e-5
end