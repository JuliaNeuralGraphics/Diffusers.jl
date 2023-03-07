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

function load_state!(fwd::FeedForward, state)
    load_state!(fwd.fn[1], state.net[1].proj)
    load_state!(fwd.fn[4], state.net[3])
end

function load_state!(block::TransformerBlock, state)
    load_state!(block.attention_1, state.attn1)
    (:attn2) in keys(state) && load_state!(block.attention_2, state.attn2)

    load_state!(block.fwd, state.ff)
    load_state!(block.norm_1, state.norm1)
    load_state!(block.norm_2, state.norm2)
    load_state!(block.norm_3, state.norm3)
end

function load_state!(tr::Transformer2D, state)
    for k in keys(state)
        load_state!(getfield(tr, k), getfield(state, k))
    end
end

function load_state!(block::ResnetBlock2D, state)
    load_state!(block.init_proj[1], state.norm1)
    load_state!(block.init_proj[2], state.conv1)
    load_state!(block.out_proj[3], state.conv2)
    load_state!(block.norm, state.norm2)

    :time_emb_proj in keys(state) && load_state!(
        block.time_emb_proj[2], state.time_emb_proj)

    (block.conv_shortcut ≡ identity) || load_state!(
        block.conv_shortcut, state.conv_shortcut)
end

load_state!(::Flux.Dropout, _) = return

load_state!(::Nothing, _) = return
