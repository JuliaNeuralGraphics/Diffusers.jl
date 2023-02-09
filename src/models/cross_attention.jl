# CrossAttention
# https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py
"""
Cross Attention Layer.
"""
struct CrossAttention
    to_q::torch.Linear
    to_k::torch.Linear
    to_v::torch.Linear
    to_out::torch.ModuleList
    added_kv_proj_dim::Union{Int,Nothing}
    group_norm
    upcast_attention::Bool
    upcast_softmax::Bool
    scale::Float32
    heads::Int
end

"""
Cross Attention Layer.

# Arguments
    query_dim: The number of channels in the query.
    cross_attention_dim: The number of channels in the encoder_hidden_states. If not given, defaults to `query_dim`.
    heads::Integer=8: The number of heads to use for multi-head attention.
    dim_head::Integer=64: The number of channels in each head.
    dropout::Float32=0.0: The dropout probability to use.
    bias::Bool=false: Set to `True` for the query, key, and value linear layers to contain a bias parameter.

# Examples
```julia
julia> CrossAttention(4; cross_attention_dim=2)
```
"""
function CrossAttention(
    query_dim::Int;
    cross_attention_dim::Union{Int,Nothing}=nothing,
    heads::Int=8,
    dim_head::Int=64,
    dropout::Float32=0.0f0,
    bias::Bool=false,
    upcast_attention::Bool=false,
    upcast_softmax::Bool=false,
    added_kv_proj_dim::Union{Int,Nothing}=nothing,
    norm_num_groups::Union{Int,Nothing}=nothing
    )

    inner_dim = dim_head * heads
    cross_attention_dim = cross_attention_dim !== nothing ? cross_attention_dim : query_dim
    scale = dim_head^(-0.5)

    if norm_num_groups !== nothing
        group_norm = Flux.GroupNorm(inner_dim, norm_num_groups)
    else
        group_norm = nothing
    end

    to_q = torch.Linear(query_dim, inner_dim, bias=bias)
    to_k = torch.Linear(cross_attention_dim, inner_dim, bias=bias)
    to_v = torch.Linear(cross_attention_dim, inner_dim, bias=bias)
    
    # Not Implemented
    @assert added_kv_proj_dim === nothing

    to_out = torch.ModuleList([torch.Linear(inner_dim, query_dim), Flux.Dropout(dropout)])

    CrossAttention(
        to_q, to_k, to_v, to_out, 
        added_kv_proj_dim, # TODO: maybe remove
        group_norm, 
        upcast_attention, upcast_softmax, 
        scale, heads
    )
end

# Forward
function (attn::CrossAttention)(
    hidden_states::AbstractArray;                             # hidden_size, sequence_length, batch_size
    encoder_hidden_states=nothing, attention_mask=nothing
)
    query = attn.to_q(hidden_states)

    encoder_hidden_states = !isnothing(encoder_hidden_states) ? encoder_hidden_states : hidden_states
    key = attn.to_k(encoder_hidden_states)
    value = attn.to_v(encoder_hidden_states)
    # attention
    hidden_states, scores = NNlib.dot_product_attention(query, key, value, nheads=attn.heads)
    hidden_states = attn.to_out[1](hidden_states)
    hidden_states = attn.to_out[2](hidden_states)
    return hidden_states
end

function load_state!(attn::CrossAttention, state)
    for k in keys(state)
        key = getfield(attn, k)
        val = getfield(state, k)
        torch.load_state!(key, val)
      end
end
