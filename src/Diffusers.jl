module Diffusers
export HGF

import MLUtils
import JSON3
import Pickle

using Adapt
using Flux
using HuggingFaceApi

const Maybe{T} = Union{Nothing, T}

const HGF = Val{:HGF}()

const FluxDeviceAdaptors = (
    Flux.FluxCPUAdaptor,
    Flux.FluxCUDAAdaptor,
    Flux.FluxAMDAdaptor)

include("feed_forward.jl")
include("attention.jl")
include("transformer.jl")
include("resnet.jl")
include("unet_2d.jl")

include("vae.jl")
include("autoencoder_kl.jl")

include("schedulers/pndm.jl")

include("load_utils.jl")

# TODO
# - UNet2DCondition: https://github.com/huggingface/diffusers/blob/c7da8fd23359a22d0df2741688b5b4f33c26df21/src/diffusers/models/unet_2d_condition.py#L249
# - get_down_block: https://github.com/huggingface/diffusers/blob/c7da8fd23359a22d0df2741688b5b4f33c26df21/src/diffusers/models/unet_2d_blocks.py#LL89C29-L89C29
# - cross attention down block: https://github.com/huggingface/diffusers/blob/c7da8fd23359a22d0df2741688b5b4f33c26df21/src/diffusers/models/unet_2d_blocks.py#L770

function main()
    down = Downsample2D(3 => 3; use_conv=true)
    up = Upsample2D(3 => 3; use_conv=true, use_transpose_conv=false)

    x = rand(Float32, 16, 16, 3, 1)
    x2 = rand(Float32, 16, 16, 320, 1)

    down_enc = SamplerBlock2D(320 => 320, Downsample2D)
    up_enc = SamplerBlock2D(320 => 320, Upsample2D)

    @show size(down(x))
    @show size(up(x))
    @show size(down_enc(x2))
    @show size(up_enc(x2))

    # TODO Encoder
    return
end

end
