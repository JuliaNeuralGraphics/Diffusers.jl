using NNlib
using Functors
using Transformers


# torch.nn.Conv2d
struct Conv2d{W<:AbstractArray, B<:Union{Nothing, AbstractArray}}
  weight::W  # WHCB
  bias::B
end

Conv2d(w) =  Conv2d(w, nothing)
_has_bias(::Conv2d{W, Nothing}) where {W<:AbstractArray}= false
_has_bias(::Conv2d) = true

function Functors.functor(::Type{<:Conv2d}, conv)
  (_has_bias(conv) ? (weight = conv.weight, bias = conv.bias) : (weight = conv.weight,)),
  y -> Conv2d(y...)
end

function (conv::Conv2d)(x::AbstractArray, cdims::ConvDims)
  # NNlib.conv(x, conv.weight, cdims) .+ conv.bias
  NNlib.conv_bias_act(x, conv.weight, cdims, conv.bias, identity)
end

function load_state!(layer::Conv2d, state)
  for k in keys(state)
    if k == :weight
      key = getfield(layer, k) # name
      val = getfield(state, k)  # tensor
      val = permutedims(val, (4, 3, 2, 1)) # BCHW -> WHCB
    elseif k == :bias
      key = getfield(layer, k) # name
      val = getfield(state, k)  # expects a single dim tensor
      val = reshape(val, (1, 1, size(val)[1], 1)) # WHCB
    else
      key = getfield(layer, nk) # name
      val = getfield(state, k)  # tensor
    end
    Transformers.HuggingFace.load_state!(key, val)
  end
end