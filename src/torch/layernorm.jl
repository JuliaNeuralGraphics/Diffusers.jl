# torch.LayerNorm
"""
    torch.LayerNorm(weight, bias)
Create a torch.LayerNorm module with weight (scale) and bias. 

# Examples
```
julia> state_dict, cfg = load_pretrained_model("runwayml/stable-diffusion-v1-5", "unet/config.json", "unet/diffusion_pytorch_model.bin")
julia> ln = torch.LayerNorm(320)
julia> torch.load_state!(ln, state_dict.down_blocks[1].attentions[1].transformer_blocks[1].norm1)
```
"""
struct LayerNorm{W<:AbstractArray, B<:Union{Nothing, AbstractArray}}
    weight::W
    bias::B
end

LayerNorm(dim::Integer; bias::Bool=true) = LayerNorm(dim, bias)
function LayerNorm(dim::Integer, bias::Bool) 
    return bias ? LayerNorm(rand(dim), rand(dim)) : LayerNorm(rand(dim), nothing)
end

function load_state!(layer::LayerNorm, state)
    for k in keys(state)
        key = getfield(layer, k)  # name
        val = getfield(state, k)  # tensor
        load_state!(key, val)
    end
end

# Forward
# https://github.com/chengchingwen/NeuralAttentionlib.jl/blob/master/src/functional/layernorm.jl
# MIT License 
function (ln::LayerNorm)(x)
    layer_norm(ln.weight, ln.bias, x)
end