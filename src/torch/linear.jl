
# torch.Linear
# Similar implementation 
# https://github.com/chengchingwen/Transformers.jl/blob/8de6eb7d3f03b94fb1077588a960803f37f08496/src/huggingface/models/base.jl#L39
# MIT License
"""
    torch.Linear(weight, bias)
Create a torch.Linear module with weight (out_features, in_features) and bias (out_features,).
Expects input data to be of size (in_features, *).

# Examples
```
julia> state_dict, cfg = load_pretrained_model("runwayml/stable-diffusion-v1-5", "unet/config.json", "unet/diffusion_pytorch_model.bin")
julia> ff = torch.Linear(rand(320, 320), rand(320, 1))
julia> torch.load_state!(ff, state_dict.down_blocks[1].attentions[1].transformer_blocks[1].attn1.to_out[1])
julia> y = ff(ones(320, 4096, 1))
```
"""
struct Linear{W<:AbstractArray, B<:Union{Nothing, AbstractArray}}
    weight::W
    bias::B
end
  
Linear(w) = Linear(w, nothing)
Linear(in::Int, out::Int; bias::Bool=true) = Linear(in, out, bias)
function Linear(in::Int, out::Int, bias::Bool) 
    bias ? Linear(rand(out, in), rand(out, 1)) : Linear(rand(out, in), nothing)
end

_has_bias(::Linear{W, Nothing}) where {W<:AbstractArray}= false
_has_bias(::Linear) = true

function Functors.functor(::Type{<:Linear}, linear)
    (_has_bias(linear) ? (weight = linear.weight, bias = linear.bias) : (weight = linear.weight,)),
    y -> Linear(y...)
end

# Forward
(l::Linear)(x::AbstractMatrix) = _has_bias(l) ? l.weight * x .+ l.bias : l.weight * x
function (l::Linear)(x::AbstractArray)
    # x is (in_features, *) y will be (out_features, *)
    old_size = size(x)
    new_size = Base.setindex(old_size, size(l.weight, 1), 1)
    new_x = reshape(x, old_size[1], :) # make it a matrix
    y = l(new_x)
    return reshape(y, new_size)
end

# load_state has been changed for bias
function load_state!(layer::Linear, state)
  for k in keys(state)
    key = getfield(layer, k)  # name
    val = getfield(state, k)  # tensor
    if k == :bias
        val = reshape(val, (size(val)[1], 1)) # bias is 1 dim, changed to (out_features, 1)
    end
    load_state!(key, val)
  end
end