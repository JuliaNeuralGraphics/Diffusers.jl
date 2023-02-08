using Diffusers: load_pretrained_model
using Transformers
using NNlib
import Diffusers.nn as nn

state_dict, cfg = load_pretrained_model("runwayml/stable-diffusion-v1-5", "unet/config.json", "unet/diffusion_pytorch_model.bin")
conv = nn.Conv2d(rand(3, 3, 4, 320), rand(1, 1, 320, 1))

# Load conv
nn.load_state!(conv, state_dict.conv_in)

# Forward 
x = ones(64, 64, 4, 1)
cdims = DenseConvDims(x, conv.weight, padding=(1,1), stride=1, dilation=1, flipkernel=true)
y = conv(x, cdims)
y[1:6, 1:6, 1, 1] # Harish: manually verified, pytorch row wise approx equals jl col wise