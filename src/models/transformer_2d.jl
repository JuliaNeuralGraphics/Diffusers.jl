"""
    Transformer2DModel
Transformer model for image-like data & continuous (actual embeddings) inputs.

When input is continuous: First, project the input (aka embedding) and reshape to emb, seq_len, batch. 
Then apply standard transformer action. Finally, reshape to image.

NOTE: Discrete case is not implemented.
"""
struct Transformer2DModel
    use_linear_projection::Bool
    num_attention_heads::Integer
    attention_head_dim::Integer
    is_input_continuous::Bool
    is_input_vectorized::Bool
    is_input_patches::Bool
    # input_continous case is implemented, rest are skipped. TODO: Add them
    in_channels::Integer
    norm::Flux.GroupNorm
    proj_in::Union{torch.Conv2d,torch.Linear}
    transformer_blocks::torch.ModuleList
    proj_out::Union{torch.Conv2d,torch.Linear}
end

"""
    Transformer2DModel 

# Arguments:
    num_attention_heads: The number of heads to use for multi-head attention.
    attention_head_dim: The number of channels in each head.
    in_channels: Pass if the input is continuous. The number of channels in the input and output.
    num_layers: The number of layers of Transformer blocks to use.
    dropout: The dropout probability to use.
    cross_attention_dim: The number of encoder_hidden_states dimensions to use.
    sample_size: The width of the latent images.
        Note that this is fixed at training time as it is used for learning a number of position embeddings. See
        `ImagePositionalEmbeddings`.
    activation_fn: Activation function to be used in feed-forward.
    attention_bias: Configure if the TransformerBlocks' attention should contain a bias parameter.

# Examples
    ```julia
    julia> Transformer2DModel(; num_attention_heads=8, attention_head_dim=40, in_channels=320, cross_attention_dim=768,)
    ```

"""
function Transformer2DModel(;
    num_attention_heads::Integer = 16,
    attention_head_dim::Integer = 88,
    in_channels::Union{Integer, Nothing} = nothing,
    out_channels::Union{Integer, Nothing} = nothing,
    num_layers::Integer = 1,
    dropout::Float32 = 0.f0,
    norm_num_groups::Integer = 32,
    cross_attention_dim::Union{Integer, Nothing} = nothing,
    attention_bias::Bool = false,
    sample_size::Union{Integer, Nothing} = nothing,
    num_vector_embeds::Union{Integer, Nothing} = nothing,
    patch_size::Union{Integer, Nothing} = nothing,
    activation_fn::String = "geglu",
    num_embeds_ada_norm::Union{Integer, Nothing} = nothing,
    use_linear_projection::Bool = false,
    only_cross_attention::Bool = false,
    upcast_attention::Bool = false,
    norm_type::String = "layer_norm",
    norm_elementwise_affine::Bool = true,
)

    inner_dim = num_attention_heads * attention_head_dim
    # 1. Transformer2DModel can process standard continous images of shape `WHCB`
    # is_input_continuous is expected to be true, as input vectorized cases are not implemented.
    is_input_continuous = (in_channels !== nothing) && (patch_size === nothing)
    if is_input_continuous == false
        error("is_input_continuous must be true. Alternate cases are not implemented.")
    end

    # 2. Define input layers
    in_channels = in_channels
    norm = Flux.GroupNorm(in_channels, norm_num_groups)
    if use_linear_projection
        proj_in = torch.Linear(in_channels, inner_dim)
    else
        # proj_in = torch.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
        weight = rand(1, 1, in_channels, inner_dim)
        bias = rand(1, 1, inner_dim, 1)
        proj_in = torch.Conv2d(weight, bias; stride=1, padding=0)
    end

    # 3. Define transformers blocks
    collection = []
    for i in 1:num_layers
        bt = BasicTransformerBlock(inner_dim, num_attention_heads, attention_head_dim;
            dropout=dropout,
            cross_attention_dim=cross_attention_dim,
            activation_fn=activation_fn,
            num_embeds_ada_norm=num_embeds_ada_norm,
            attention_bias=attention_bias,
            only_cross_attention=only_cross_attention,
            upcast_attention=upcast_attention,
            norm_type=norm_type,
            norm_elementwise_affine=norm_elementwise_affine,
        )
        append!(collection, [bt])
    end
    transformer_blocks = torch.ModuleList(collection)

    # 4. Define output layers
    if is_input_continuous
        # TODO: should use out_channels for continous projections
        if use_linear_projection
            proj_out = torch.Linear(in_channels, inner_dim)
        else
            # proj_out = torch.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)
            weight = rand(1, 1, inner_dim, in_channels)
            bias = rand(1, 1, in_channels, 1)
            proj_out = torch.Conv2d(weight, bias; stride=1, padding=0)
        end
    end
    is_input_vectorized = false
    is_input_patches = false
    Transformer2DModel(use_linear_projection, num_attention_heads, attention_head_dim, 
        is_input_continuous, is_input_vectorized, is_input_patches, in_channels, norm,
        proj_in, transformer_blocks, proj_out)
end

# TODO: add types
function (t2dm::Transformer2DModel)(
    hidden_states;
    encoder_hidden_states=nothing,
    timestep=nothing,
    class_labels=nothing,
    cross_attention_kwargs=nothing,
    return_dict::Bool=true,
)
    # TODO: is_input_continuous is assumed to be true
    # 1. Input
    # batch, _, height, width = size(hidden_states)
    width, height, channels, batch = size(hidden_states)
    residual = hidden_states
    hidden_states = t2dm.norm(hidden_states)
    if (! t2dm.use_linear_projection) # proj_in is a conv
        hidden_states = t2dm.proj_in(hidden_states) # WHCB
        hidden_states = reshape(hidden_states, (width*height, channels, batch))
        hidden_states = permutedims(hidden_states, (2, 1, 3)) # channels, width*height, batch
        inner_dim = channels
    else # proj_in is a linear layer
        inner_dim = channels
        hidden_states = reshape(hidden_states, (width*height, channels, batch))
        hidden_states = permutedims(hidden_states, (2, 1, 3)) # channels, width*height, batch
        hidden_states = t2dm.proj_in(hidden_states)
    end

    # 2. Blocks
    for (i, block) in enumerate(t2dm.transformer_blocks)
        hidden_states = block(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            cross_attention_kwargs=cross_attention_kwargs,
            class_labels=class_labels,
        )
    end

    if t2dm.is_input_continuous
        if ! t2dm.use_linear_projection
            hidden_states = permutedims(hidden_states, (2, 1, 3)) # width*height, channels, batch
            hidden_states = reshape(hidden_states, (width, height, channels, batch))
            hidden_states = t2dm.proj_out(hidden_states)
        else
            hidden_states = t2dm.proj_out(hidden_states)
            hidden_states = permutedims(hidden_states, (2, 1, 3)) # width*height, channels, batch
            hidden_states = reshape(hidden_states, (width, height, channels, batch))
        end
    end
    output = hidden_states + residual
    return (sample=output,)
end

function load_state!(layer::Transformer2DModel, state)
    for k in keys(state)
        key = getfield(layer, k)
        val = getfield(state, k)
        load_state!(key, val)
      end
end
