"""
Supports loading PyTorch trained modules and do forward in julia. 
"""
module torch

using NNlib
using Functors
using Transformers

export 
    Conv2d

include("conv.jl")
end