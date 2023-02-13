# https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention.py#L405
struct GEGLU
    proj::torch.Linear
end

GEGLU(dim_in::Integer, dim_out::Integer) = GEGLU(torch.Linear(dim_in, dim_out*2))
Functors.functor(::Type{<:GEGLU}, geglu) = (proj = geglu.proj,), y -> GEGLU(y...)

function (geglu::GEGLU)(hidden_states)
    hidden_states = geglu.proj(hidden_states)
    hidden_states, gate = MLUtils.chunk(hidden_states, 2, dims=1)
    return NNlib.gelu(gate) .* hidden_states
end

function load_state!(layer::GEGLU, state)
    for k in keys(state)
        key = getfield(layer, k)  # name
        val = getfield(state, k)  # tensor
        torch.load_state!(key, val)
    end
end

struct FeedForward
    net::torch.ModuleList
end

function FeedForward(dim::Integer; 
    dim_out::Union{Nothing, Integer}=nothing,
    mult::Integer=4,
    dropout::Float32=0.f0,
    activation_fn::String="geglu",
    final_dropout::Bool=false,
)
    inner_dim = Int32(dim * mult)
    dim_out = dim_out !== nothing ? dim_out : dim

    if activation_fn == "geglu"
        act_fn = GEGLU(dim, inner_dim)
    else
        error("Not Implemented")
    end

    net = [act_fn, Flux.Dropout(dropout), torch.Linear(inner_dim, dim_out)]
    if final_dropout
        net = append!(net, Flux.Dropout(dropout))
    end
    FeedForward(torch.ModuleList(net))
end

function load_state!(layer::FeedForward, state)
    for k in keys(state)
        key = getfield(layer, k)  # name
        val = getfield(state, k)  # tensor
        torch.load_state!(key, val)
    end
end

function (ff::FeedForward)(hidden_states)
    for (i, netᵢ) in enumerate(ff.net)
       hidden_states = netᵢ(hidden_states) 
    end
    return hidden_states
end