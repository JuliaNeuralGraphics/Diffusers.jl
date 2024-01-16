module Diffusers

import JSON3
import MLUtils
import Pickle

using Adapt
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

function main(
    prompts::Vector{String};
    n_inference_steps::Int = 20,
    n_images_per_prompt::Int = 1,
    precision = f16, device = gpu,
)
    GC.gc(false)
    GC.gc(true)

    sd = StableDiffusion("runwayml/stable-diffusion-v1-5") |> f16 |> gpu
    println("Running StableDiffusion on $(get_backend(sd))")

    images = sd(prompts; n_images_per_prompt, n_inference_steps)

    idx = 1
    for prompt in prompts, i in 1:n_images_per_prompt
        joined_prompt = replace(prompt, ' ' => '-')
        save("$joined_prompt-$i.png", rotr90(RGB{N0f8}.(images[:, :, idx])))
        idx += 1
    end
    return
end

function main_clip()
    GC.gc(false)
    GC.gc(true)

    sd = StableDiffusion("runwayml/stable-diffusion-v1-5") |> f16 |> gpu
    println("Running StableDiffusion on $(get_backend(sd))")

    prompts = ["ancient house", "modern house"]
    clip(sd, prompts; n_inference_steps=10, n_interpolation_steps=120)
    return
end

end
