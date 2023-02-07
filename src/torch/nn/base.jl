using NNlib

# torch.nn.Conv2d
struct Conv2d{W<:AbstractArray, B<:Union{Nothing, AbstractArray}}
    weight::W
    bias::B
end

Conv2d(w) =  Conv2d(w, nothing)
_has_bias(::Conv2d{W, Nothing}) where {W<:AbstractArray}= false
_has_bias(::Conv2d) = true

function Functors.functor(::Type{<:Conv2d}, conv)
  (_has_bias(conv) ? (weight = conv.weight, bias = conv.bias) : (weight = conv.weight,)),
  y -> Conv2d(y...)
end

function (conv::Conv2d)(x::AbstractArray) 
    NNlib.conv(x, conv.weight; stride=1, pad=(1,1), dilation=1, flipped=false, groups=1)
    # _has_bias(l) ? l.weight * x .+ l.bias : l.weight * x
end

c = Conv2d(rand(3, 3, 4, 5))
res = c(rand(64, 64, 4, 1))

size(c.weight)
size(res)
c = Conv2d(rand(320, 4, 3, 3), rand(320))
c = Conv2d(rand(3, 3, 4, 340))

size(state_dict.conv_in.weight)

state_dict, cfg = load_pretrained_model("runwayml/stable-diffusion-v1-5", "unet/config.json", "unet/diffusion_pytorch_model.bin")
Transformers.HuggingFace.load_state!(c, state_dict.conv_in)

# function (l::FakeTHLinear)(x::AbstractArray)
#   old_size = size(x)
#   new_size = Base.setindex(old_size, size(l.weight, 1), 1)
#   new_x = reshape(x, old_size[1], :)
#   y = l(new_x)
#   return reshape(y, new_size)
# end