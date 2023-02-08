"""
Supports loading PyTorch trained modules and do forward in julia. 
"""
module torch

using NNlib
using Functors
using Transformers

export 
    Conv2d,
    Linear

include("conv.jl")
include("linear.jl")
end