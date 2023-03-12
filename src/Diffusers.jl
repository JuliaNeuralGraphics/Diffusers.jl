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

include("schedulers/pndm.jl")

include("load_utils.jl")

# TODO
# - UNet2DCondition: https://github.com/huggingface/diffusers/blob/c7da8fd23359a22d0df2741688b5b4f33c26df21/src/diffusers/models/unet_2d_condition.py#L249
# - get_down_block: https://github.com/huggingface/diffusers/blob/c7da8fd23359a22d0df2741688b5b4f33c26df21/src/diffusers/models/unet_2d_blocks.py#LL89C29-L89C29
# - cross attention down block: https://github.com/huggingface/diffusers/blob/c7da8fd23359a22d0df2741688b5b4f33c26df21/src/diffusers/models/unet_2d_blocks.py#L770

end
