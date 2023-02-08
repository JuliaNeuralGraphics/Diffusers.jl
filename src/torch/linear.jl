
# torch.Linear
# Similar implementation https://github.com/chengchingwen/Transformers.jl/blob/master/src/huggingface/models/base.jl#L39
struct Linear{W<:AbstractArray, B<:Union{Nothing, AbstractArray}}
	weight::W
	bias::B
end
  
Linear(w) = Linear(w, nothing)

_has_bias(::Linear{W, Nothing}) where {W<:AbstractArray}= false
_has_bias(::Linear) = true

function Functors.functor(::Type{<:Linear}, linear)
	(_has_bias(linear) ? (weight = linear.weight, bias = linear.bias) : (weight = linear.weight,)),
	y -> Linear(y...)
end

(l::Linear)(x::AbstractMatrix) = _has_bias(l) ? l.weight * x .+ l.bias : l.weight * x

function (l::Linear)(x::AbstractArray)
    # x is (in_features, *) y will be (out_features, *)
    old_size = size(x)
    new_size = Base.setindex(old_size, size(l.weight, 1), 1)
    new_x = reshape(x, old_size[1], :) # make it a matrix
    y = l(new_x)
    return reshape(y, new_size)
end

function load_state!(layer::Linear, state)
  for k in keys(state)
    key = getfield(layer, k)  # name
    val = getfield(state, k)  # tensor
    if k == :bias
        val = reshape(val, (size(val)[1], 1)) # bias added to out_channels
    end
    Transformers.HuggingFace.load_state!(key, val)
  end
end