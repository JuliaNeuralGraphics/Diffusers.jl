"""
Supports loading PyTorch trained modules and do forward in julia. 
"""
module torch

using Flux
using NNlib
using Functors
using Transformers

export 
    Conv2d,
    Linear,
    ModuleList

load_state!(layer::Any, state::Any) = Transformers.HuggingFace.load_state!(layer, state)

include("conv.jl")
include("linear.jl")
include("modulelist.jl")
end