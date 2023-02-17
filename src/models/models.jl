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

    # All models in `models` module must call models.load_state!
    # Order: models.load_state! -> torch.load_state! -> Transformers.load_state!
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