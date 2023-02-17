function load_state!(layer::GroupNorm, state)
    layer.γ = state.weight
    layer.β = state.bias
    return nothing
end
