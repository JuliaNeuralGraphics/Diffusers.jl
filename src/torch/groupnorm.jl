function load_state!(layer::GroupNorm, state) # uses Flux.GroupNorm
    layer.γ = state.weight
    layer.β = state.bias
    return nothing
end
