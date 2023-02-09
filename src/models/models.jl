"""
    Support models defined in diffusers library.
"""
module models

    using Flux
    using NNlib
    using Functors
    using Transformers
    using ..torch

    export
    CrossAttention

    include("cross_attention.jl")
end