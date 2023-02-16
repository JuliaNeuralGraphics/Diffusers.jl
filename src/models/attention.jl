# https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention.py#L405
"""
    GEGLU
A variant of the gated linear unit activation function from https://arxiv.org/abs/2002.05202.
"""
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
        load_state!(key, val)
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
        error("$activation_fn is not implemented. GEGLU is available.")
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
        load_state!(key, val)
    end
end

function (ff::FeedForward)(hidden_states)
    for (i, netᵢ) in enumerate(ff.net)
       hidden_states = netᵢ(hidden_states) 
    end
    return hidden_states
end


"""
    BasicTransformerBlock(dim::Integer, num_attention_heads::Integer, attention_head_dim::Integer)
A basic Transformer block.

# Examples
```
julia> tb = BasicTransformerBlock(320, 8, 40; cross_attention_dim=768)
```

"""
struct BasicTransformerBlock
    attn1::CrossAttention
    attn2::CrossAttention
    ff::FeedForward
    norm1::torch.LayerNorm
    norm2::torch.LayerNorm
    norm3::torch.LayerNorm
    only_cross_attention::Bool
end

function BasicTransformerBlock(
    dim::Integer, num_attention_heads::Integer, attention_head_dim::Integer;
    dropout=0.0f0, cross_attention_dim::Union{Integer,Nothing}=nothing,
    activation_fn::String="geglu", 
    num_embeds_ada_norm::Union{Nothing,Integer}=nothing,
    attention_bias::Bool=false, only_cross_attention::Bool=false,
    upcast_attention::Bool=false, norm_elementwise_affine::Bool=true,
    norm_type::String="layer_norm", final_dropout::Bool=false,
)
    use_ada_layer_norm_zero = (num_embeds_ada_norm !== nothing && norm_type == "ada_norm_zero")
    use_ada_layer_norm = (num_embeds_ada_norm !== nothing && norm_type == "ada_norm")
    if use_ada_layer_norm || use_ada_layer_norm_zero
        error("AdaLayerNorm or AdaLayerNormZero is not implemented. torch.LayerNorm is available.")
    end
    if use_ada_layer_norm || use_ada_layer_norm_zero
        error("AdaLayerNorm or AdaLayerNormZero is not implemented. torch.LayerNorm is available.")
    end

    # 1. Self-Attn
    attn1 = CrossAttention(dim;
        heads=num_attention_heads,
        dim_head=attention_head_dim,
        dropout=dropout,
        bias=attention_bias,
        cross_attention_dim=(only_cross_attention ? cross_attention_dim : nothing),
        upcast_attention=upcast_attention,
    )

    ff = FeedForward(dim; dropout=dropout, activation_fn=activation_fn, final_dropout=final_dropout)

    # 2. Cross-Attn
    if cross_attention_dim !== nothing
        attn2 = CrossAttention(dim;
            cross_attention_dim=cross_attention_dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            upcast_attention=upcast_attention,
        )  # is self-attn if encoder_hidden_states is none
    else
        attn2 = nothing
    end
    if norm_elementwise_affine === false
        error("layernom with norm_elementwise_affine=$norm_elementwise_affine is not implemented.")
    end
    norm1 = torch.LayerNorm(dim)
    
    if cross_attention_dim !== nothing
        # We currently only use AdaLayerNormZero for self attention where there will only be one attention block.
        # I.e. the number of returned modulation chunks from AdaLayerZero would not make sense if returned during
        # the second cross attention block.
        
        # TODO: add AdaLayerNorm based on use_ada_layer_norm
        norm2 = torch.LayerNorm(dim)
    else
        norm2 = nothing
    end

    # 3. Feed-forward
    norm3 = torch.LayerNorm(dim)

    return BasicTransformerBlock(attn1, attn2, ff, norm1, norm2, norm3, only_cross_attention)
end

# Forward
function (tb::BasicTransformerBlock)(hidden_states;
    encoder_hidden_states=nothing,
    timestep=nothing,
    attention_mask=nothing,
    cross_attention_kwargs=nothing,
    class_labels=nothing,
)
    # TODO: add AdaLayerNorm
    norm_hidden_states = tb.norm1(hidden_states)
    cross_attention_kwargs = cross_attention_kwargs !== nothing ? cross_attention_kwargs : Dict()
    # attn1_encoder_hidden_states = 

    # 1. Self-Attention
    attn_output = tb.attn1(norm_hidden_states;
        encoder_hidden_states=(tb.only_cross_attention ? encoder_hidden_states : nothing),
        attention_mask=attention_mask,
        cross_attention_kwargs...,
    )
    hidden_states = attn_output + hidden_states

    if tb.attn2 !== nothing
        norm_hidden_states = tb.norm2(hidden_states)
        # 2. Cross-Attention
        attn_output = tb.attn2(norm_hidden_states;
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            cross_attention_kwargs...,
        )
        hidden_states = attn_output + hidden_states
    end

    # 3. Feed-forward
    norm_hidden_states = tb.norm3(hidden_states)
    ff_output = tb.ff(norm_hidden_states)
    hidden_states = ff_output + hidden_states

    return hidden_states
end

function load_state!(layer::BasicTransformerBlock, state)
    for k in keys(state)
        key = getfield(layer, k)  # name
        val = getfield(state, k)  # tensor
        load_state!(key, val)
    end
end