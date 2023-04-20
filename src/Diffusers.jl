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

using AMDGPU
using KernelAbstractions
const Backend = ROCBackend()

function sync_free!(args...)
    KernelAbstractions.synchronize(Backend)
    KernelAbstractions.unsafe_free!.(args)
end

const Maybe{T} = Union{Nothing, T}

# TODO better way of handling this
const FluxDeviceAdaptors = (
    Flux.FluxCPUAdaptor,
    Flux.FluxCUDAAdaptor,
    Flux.FluxAMDAdaptor)
const FluxEltypeAdaptors = (
    Flux.FluxEltypeAdaptor{Float32},
    Flux.FluxEltypeAdaptor{Float16})

get_pb(n, desc::String) = Progress(
    n; desc, dt=1, barglyphs=BarGlyphs("[=> ]"), barlen=50, color=:white)

# TODO
# This matches what PyTorch is doing.
# Upstream to Flux.
# Currently it is doing: (x - μ) / (σ + ϵ)
# But instead it should: (x - μ) / sqrt(σ² + ϵ)
function (ln::LayerNorm)(x::AbstractArray)
    ϵ = convert(float(eltype(x)), ln.ϵ)
    μ, σ² = _normalize(x; dims=1:length(ln.size))
    y = ln.diag((x .- μ) .* inv.(sqrt.(σ² .+ ϵ)))
    sync_free!(μ, σ²)
    return y
end

function (gn::Flux.GroupNorm)(x::AbstractArray)
    sz = size(x)
    x2 = reshape(x, sz[1:end - 2]..., sz[end - 1] ÷ gn.G, gn.G, sz[end])
    N = ndims(x2) # == ndims(x)+1
    reduce_dims = 1:(N - 2)
    affine_shape = ntuple(i -> i ∈ (N - 1, N - 2) ? size(x2, i) : 1, N)

    μ, σ² = _normalize(x2; dims=reduce_dims)
    γ = reshape(gn.γ, affine_shape)
    β = reshape(gn.β, affine_shape)

    ϵ = convert(float(eltype(x)), gn.ϵ)
    scale = γ .* inv.(sqrt.(σ² .+ ϵ))
    bias = -scale .* μ .+ β

    sync_free!(μ, σ²)
    return reshape(gn.λ.(scale .* x2 .+ bias), sz)
end

function _normalize(x::AbstractArray{Float16}; dims)
    x_fp32 = Float32.(x)
    μ, σ² = _normalize(x_fp32; dims)
    m, v = Float16.(μ), Float16.(σ²)
    sync_free!(x_fp32, μ, σ²)
    return m, v
end

function _normalize(x; dims)
    μ = mean(x; dims)
    σ² = var(x; dims, mean=μ, corrected=false)
    μ, σ²
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

function mm()
    sd = StableDiffusion("runwayml/stable-diffusion-v1-5") |> f32 |> cpu
    println("Running StableDiffusion on $(get_backend(sd))")

    prompts = ["wooden cat"]
    images = sd(prompts; n_images_per_prompt=1, n_inference_steps=10)
    for i in 1:size(images, 3)
        save("image-$i.png", rotr90(RGB{N0f8}.(images[:, :, i])))
    end
    return
end

function main()
    m = LayerNorm(320)
    lf = m |> f32 |> gpu
    lh = m |> f16 |> gpu

    x = rand(Float32, 320, 4096, 1)
    xf = x |> f32 |> gpu
    xh = x |> f16 |> gpu

    y = m(x)
    yf = lf(xf) |> cpu
    yh = lh(xh) |> cpu |> f32

    println()
    @show sum(y)
    @show sum(yf)
    @show sum(yh)
    return
end

end
