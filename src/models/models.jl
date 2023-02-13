"""
    Support models defined in diffusers library.
"""
module models

    using Flux
    using NNlib
    using MLUtils
    using Functors
    using Transformers
    using ..torch

    export
    CrossAttention,
    GEGLU

    include("cross_attention.jl")
    include("attention.jl")
end