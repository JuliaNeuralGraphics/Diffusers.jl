# TODO verify keys before loop and fail early

function load_state!(layer::Flux.Conv, state)
    for k in keys(state)
        val = if k == :weight
            # BCHW -> WHCB & flip kernel from cross-correlation to convolution.
            w = permutedims(getfield(state, k), (4, 3, 2, 1))
            w = w[end:-1:1, end:-1:1, :, :]
        else
            getfield(state, k)
        end
        getfield(layer, k) .= val
    end
end

function load_state!(layer::Flux.Dense, state)
    for k in keys(state)
        getfield(layer, k) .= getfield(state, k)
    end
end

function load_state!(chain::Flux.Chain, state)
    layer_id = 1
    for layer in chain
        (layer isa Dropout || layer ≡ identity) && continue
        load_state!(layer, state[layer_id])
        layer_id += 1
    end
end

load_state!(::Flux.Dropout, _) = return

function load_state!(layer::Flux.LayerNorm, state)
    for k in keys(state)
        key = getfield(layer.diag, k == :weight ? :scale : k)
        val = getfield(state, k)
        key .= val
    end
end

function load_state!(layer::Flux.GroupNorm, state)
    layer.γ = state.weight
    layer.β = state.bias
    return nothing
end

function load_state!(attn::CrossAttention, state)
    for k in keys(state)
        load_state!(getfield(attn, k), getfield(state, k))
    end
end
