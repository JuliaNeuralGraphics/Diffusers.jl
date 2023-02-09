using Diffusers
using Test
using Diffusers: load_pretrained_model
using Transformers
using NNlib
using Diffusers.torch
using Flux

using Diffusers.models

c = CrossAttention(3; heads=8)
size(c(ones(3, 2, 1))[1])