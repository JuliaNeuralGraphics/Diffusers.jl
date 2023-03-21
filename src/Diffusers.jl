module Diffusers
export HGF

import MLUtils
import JSON3
import Pickle

using Adapt
using Flux
using HuggingFaceApi
using OrderedCollections

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
include("clip/basic.jl")
include("clip/tokenizer.jl")
include("clip/models.jl")

include("load_utils.jl")

# >>> vae.encoder(x).sum()
# tensor(28057.0957, grad_fn=<SumBackward0>)

function main()
    kl = AutoencoderKL(
        "runwayml/stable-diffusion-v1-5";
        state_file="vae/diffusion_pytorch_model.bin",
        config_file="vae/config.json")
    x = ones(Float32, 256, 256, 3, 1)

    @show sum(kl.encoder(x))
    @show sum(kl(x))

    # y = kl(x)
    # @show size(y)
    # @show sum(y)

    # y = kl(x; sample_posterior = true)
    # @show size(y)
    # @show sum(y)

    return
end

function tk()
    input_texts = [
        "Hello, world!",
        "There is nothing basically... I mean it quite literally",
        "I was now on a dark path, unsettled by a future filled with big data and small comprehension.",
    ]
    println("Input texts:")
    display(input_texts); println()

    tokenizer = CLIPTokenizer()
    tokens, pad_mask = tokenize(tokenizer, input_texts; context_length=32)
    println("Tokens:")
    display(tokens); println()
    display(pad_mask); println()
    @show size(pad_mask)

    texts = [
        decode(tokenizer, @view(tokens[:, i]))
        for i in 1:size(tokens, 2)]
    println("Decoded texts:")
    display(texts); println()

    nothing
end

"""
- clip text model: https://github.com/huggingface/transformers/blob/fb366b9a2a94b38171896f6ba9fb9ae8bffd77af/src/transformers/models/clip/modeling_clip.py#L769
- CLIPFeatureExtractor: https://github.com/huggingface/transformers/blob/fb366b9a2a94b38171896f6ba9fb9ae8bffd77af/src/transformers/models/clip/feature_extraction_clip.py#L26
"""

function ttt()
    transformer = CLIPTextTransformer(
        "runwayml/stable-diffusion-v1-5";
        state_file="text_encoder/pytorch_model.bin",
        config_file="text_encoder/config.json")

    # x = Int32[1; 2;; 5; 6;; 49407; 49408;;]
    # x = Int32[1; 2;;]
    # y = transformer(x)
    # @show size(y)
    # @show sum(y)

    x = ones(Float32, 768, 2, 1)
    for i in 1:length(transformer.encoder.layers)
        @show sum(transformer.encoder.layers[i](x))
    end

    return
end

end
