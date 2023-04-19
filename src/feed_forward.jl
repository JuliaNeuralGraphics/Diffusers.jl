function geglu(x)
    h, gate = MLUtils.chunk(x, 2; dims=1)
    gate = gelu(gate)
    y = h .* gate
    sync_free!(gate)
    return y
end

struct FeedForward{F}
    fn::F
end
Flux.@functor FeedForward

# NOTE no final dropout
function FeedForward(;
    dim::Int, dim_out::Maybe{Int} = nothing, dim_multiplier::Int = 4,
    dropout::Real = 0,
)
    inner_dim = dim * dim_multiplier
    dim_out = isnothing(dim_out) ? dim : dim_out
    FeedForward(Chain(
        Dense(dim => inner_dim * 2), geglu,
        iszero(dropout) ? identity : Dropout(dropout),
        Dense(inner_dim => dim_out)))
end

(fwd::FeedForward)(x) = fwd.fn(x)
