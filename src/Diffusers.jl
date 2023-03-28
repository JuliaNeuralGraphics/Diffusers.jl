module Diffusers

import MLUtils
import JSON3
import Pickle

using Statistics
using Adapt
using Flux
using HuggingFaceApi
using OrderedCollections
using ImageIO
using FileIO
using ImageCore
using ProgressMeter

const Maybe{T} = Union{Nothing, T}

const FluxDeviceAdaptors = (
    Flux.FluxCPUAdaptor,
    Flux.FluxCUDAAdaptor,
    Flux.FluxAMDAdaptor)

get_pb(n, desc::String) = Progress(
    n; desc, dt=1, barglyphs=BarGlyphs("[=> ]"), barlen=50, color=:white)

# TODO
# This matches what PyTorch is doing.
# Upstream to Flux.
# Currently it is doing: (x - μ) / (σ + ϵ)
# But instead it should: (x - μ) / sqrt(σ² + ϵ)
function (ln::LayerNorm)(x::AbstractArray)
    ϵ = convert(float(eltype(x)), ln.ϵ)
    μ = mean(x; dims=1:length(ln.size))
    σ² = var(x; dims=1:length(ln.size), mean=μ, corrected=false)
    ln.diag((x .- μ) ./ sqrt.(σ² .+ ϵ))
end

function (gn::Flux.GroupNorm)(x::AbstractArray)
    sz = size(x)
    x2 = reshape(x, sz[1:end-2]..., sz[end-1]÷gn.G, gn.G, sz[end])
    N = ndims(x2)  # == ndims(x)+1
    reduce_dims = 1:N-2
    affine_shape = ntuple(i -> i ∈ (N-1, N-2) ? size(x2, i) : 1, N)
    μ = mean(x2; dims=reduce_dims)
    σ² = var(x2; dims=reduce_dims, mean=μ, corrected=false)

    γ = reshape(gn.γ, affine_shape)
    β = reshape(gn.β, affine_shape)

    scale = γ ./ sqrt.(σ² .+ gn.ϵ)
    bias = -scale .* μ .+ β
    return reshape(gn.λ.(scale .* x2 .+ bias), sz)
end

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
    kl = AutoencoderKL(
        "runwayml/stable-diffusion-v1-5";
        state_file="vae/diffusion_pytorch_model.bin",
        config_file="vae/config.json")

    x = ones(Float32, 256, 256, 3, 1)
    y = kl.encoder(x)
    @show sum(y)
    @show size(y)

    # y = kl(x)
    # @show size(y)
    # @show sum(y)

    # y = kl(x; sample_posterior = true)
    # @show size(y)
    # @show sum(y)
    return
end

function mm()
    sd = StableDiffusion("runwayml/stable-diffusion-v1-5")
    images = sd([
        "diamond forest",
        "wooden cat",
    ]; n_images_per_prompt=2)
    @show size(images)
    for i in 1:size(images, 3)
        save("image-$i.png", rotr90(RGB{N0f8}.(images[:, :, i])))
    end
    return
end

end
