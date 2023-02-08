# Similar to torch.conv2d, but runs in WHCB
"""
    torch.Conv2d(weight, bias)
Create a conv module in WHCB format.

# Examples
```
julia> state_dict, cfg = Diffusers.load_pretrained_model("runwayml/stable-diffusion-v1-5", "unet/config.json", "unet/diffusion_pytorch_model.bin")
julia> conv = torch.Conv2d(rand(3, 3, 4, 320), rand(1, 1, 320, 1)) 
julia> torch.load_state!(conv, state_dict.conv_in) # state_dict.conv_in has BCHW weights, converts to WHCB
julia> cdims = DenseConvDims(ones(64, 64, 4, 1), conv.weight, padding=(1,1), stride=1, dilation=1, flipkernel=true)
julia> conv(ones(64, 64, 4, 1), cdims)
```
"""
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

# Forward
function (conv::Conv2d)(x::AbstractArray, cdims::ConvDims)
    # NNlib.conv(x, conv.weight, cdims) .+ conv.bias
    NNlib.conv_bias_act(x, conv.weight, cdims, conv.bias, identity)
end

function load_state!(layer::Conv2d, state)
    for k in keys(state)
    if k == :weight
        key = getfield(layer, k)  # name
        val = getfield(state, k)  # tensor
        # (out_channels, in_channels​, kernel_size[0], kernel_size[1]) -> 
        # (kernel_size[1], kernel_size[0], in_channels​, out_channels)
        val = permutedims(val, (4, 3, 2, 1)) # BCHW -> WHCB
    elseif k == :bias
        key = getfield(layer, k)  # name
        val = getfield(state, k)  # expects a single dim tensor
        val = reshape(val, (1, 1, size(val)[1], 1)) # bias added to out_channels
    else
        key = getfield(layer, nk) # name
        val = getfield(state, k)  # tensor
    end
    Transformers.HuggingFace.load_state!(key, val)
    end
end