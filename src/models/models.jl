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
    BasicTransformerBlock,
    Transformer2DModel

    # Diffusion models call load_state! inside torch, which eventually calls
    # Transformers.load_state! when there no hits with torch/models types.
    # So, use load_state! to load models within all Diffusers.models
    load_state!(layer::Any, state::Any) = torch.load_state!(layer, state)

    # HACK: When ModuleList have modules from `models`, it needs to use the
    # load_state! defined in models.
    function load_state!(layer::ModuleList, state)
        for (i, layerᵢ) in enumerate(layer)
            # TODO: handle cases where layer is missing
            if (typeof(layerᵢ) <: Dropout)
                println("WARN: Flux.Dropout is not loaded as there are no parameters in the state_dict.")
            else
                load_state!(layerᵢ, state[i]) 
            end
        end
    end

    include("cross_attention.jl")
    include("attention.jl")
    include("transformer_2d.jl")
end