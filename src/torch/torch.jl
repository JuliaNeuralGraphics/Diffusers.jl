"""
Supports loading PyTorch trained modules and do forward in julia. 
"""
module torch

using Flux:Dropout,GroupNorm
using NNlib
using Functors
using Transformers
using NeuralAttentionlib:layer_norm
export 
    Conv2d,
    Linear,
    ModuleList,
    LayerNorm

load_state!(layer::Any, state::Any) = Transformers.HuggingFace.load_state!(layer, state)

include("conv.jl")
include("linear.jl")
include("modulelist.jl")
include("layernorm.jl")
include("groupnorm.jl")
end