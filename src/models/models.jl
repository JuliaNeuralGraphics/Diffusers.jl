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
    GEGLU,
    FeedForward,
    BasicTransformerBlock

    # Diffusion models call load_state! inside torch, which eventually calls
    # Transformers.load_state! when there no hits with torch/models types.
    # So, use load_state! to load models within all Diffusers.models
    load_state!(layer::Any, state::Any) = torch.load_state!(layer, state)
    
    include("cross_attention.jl")
    include("attention.jl")
end