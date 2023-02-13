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

