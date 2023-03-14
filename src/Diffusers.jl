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

function main()
    kl = AutoencoderKL(
        "runwayml/stable-diffusion-v1-5";
        weights_file="",
        config_file="vae/config.json")

    return
end

end
