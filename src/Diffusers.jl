module Diffusers

import JSON3
import MLUtils
import Pickle

using Adapt
using AMDGPU
using FileIO
using Flux
using HuggingFaceApi
using ImageCore
using ImageIO
using OrderedCollections
using ProgressMeter
using Statistics
using VideoIO

const Maybe{T} = Union{Nothing, T}

get_pb(n, desc::String) = Progress(
    n; desc, dt=1, barglyphs=BarGlyphs("[=> ]"), barlen=50, color=:white)

include("timestep.jl")
include("feed_forward.jl")
include("attention.jl")
include("transformer.jl")
include("resnet.jl")

include("unet/blocks.jl")
include("unet/2d_condition.jl")

include("autoencoder/blocks.jl")
include("autoencoder/kl.jl")

include("clip/basic.jl")
include("clip/tokenizer.jl")
include("clip/models.jl")

include("schedulers/pndm.jl")
include("stable_diffusion.jl")

include("load_utils.jl")

function main()
    GC.gc()
    AMDGPU.HIP.device_synchronize()
    AMDGPU.HIP.reclaim()

    sd = StableDiffusion("runwayml/stable-diffusion-v1-5") |> f16 |> gpu
    println("Running StableDiffusion on $(get_backend(sd))")

    n_images_per_prompt = 1
    prompts = ["painting of a farmer in the field"]
    images = sd(prompts; n_images_per_prompt, n_inference_steps=20)

    idx = 1
    for prompt in prompts, i in 1:n_images_per_prompt
        joined_prompt = replace(prompt, ' ' => '-')
        save("$joined_prompt-$i.png", rotr90(RGB{N0f8}.(images[:, :, idx])))
        idx += 1
    end
    return
end

function main_clip()
    GC.gc()
    AMDGPU.HIP.device_synchronize()
    AMDGPU.HIP.reclaim()

    sd = StableDiffusion("runwayml/stable-diffusion-v1-5") |> f16 |> gpu
    println("Running StableDiffusion on $(get_backend(sd))")

    prompts = ["ancient house", "modern house"]
    clip(sd, prompts; n_inference_steps=10, n_interpolation_steps=120)
    return
end

end
