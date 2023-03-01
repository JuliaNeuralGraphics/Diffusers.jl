struct ResnetBlock2D
    pre_norm::Bool
    in_channels::Integer
    out_channels::Integer
    use_conv_shortcut::Bool
    time_embedding_norm::String
    output_scale_factor::Integer
    norm1::Flux.GroupNorm
    conv1::torch.Conv2d
    time_emb_proj::Union{Nothing,torch.Linear}
    norm2::Flux.GroupNorm
    dropout::Flux.Dropout
    conv2::torch.Conv2d
    nonlinearity::String
    use_in_shortcut::Bool
    conv_shortcut::Union{Nothing,torch.Conv2d}
end

function ResnetBlock2D(in_channels;
    out_channels=nothing,
    conv_shortcut=false,
    dropout=0.0,
    temb_channels=512,
    groups=32,
    groups_out=nothing,
    pre_norm=true,
    eps=1e-6,
    non_linearity="swish",
    time_embedding_norm="default",
    kernel=nothing,
    output_scale_factor=1.0,
    use_in_shortcut=nothing,
    up=false,
    down=false,
)
    pre_norm = true
    out_channels = out_channels === nothing ? in_channels : out_channels
    use_conv_shortcut = conv_shortcut
    if groups_out === nothing
        groups_out = groups
    end
    norm1 = Flux.GroupNorm(in_channels, groups)

    # conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
    weight = rand(3, 3, in_channels, out_channels)
    bias = rand(1, 1, out_channels, 1)
    conv1 = torch.Conv2d(weight, bias; stride=1, padding=1)
    
    time_emb_proj = nothing
    if temb_channels !== nothing
        if time_embedding_norm == "default"
            time_emb_proj_out_channels = out_channels
        elseif time_embedding_norm == "scale_shift"
            time_emb_proj_out_channels = out_channels * 2
        else
            error("unknown time_embedding_norm : $time_embedding_norm")
        end
        time_emb_proj = torch.Linear(temb_channels, time_emb_proj_out_channels)    
    end

    norm2 = Flux.GroupNorm(out_channels, groups_out)
    dropout = Flux.Dropout(dropout)
    # conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
    weight = rand(3, 3, out_channels, out_channels)
    bias = rand(1, 1, out_channels, 1)
    conv2 = torch.Conv2d(weight, bias; stride=1, padding=1)

    # down & up cases are not implemented
    use_in_shortcut = use_in_shortcut === nothing ? in_channels != out_channels : use_in_shortcut
    conv_shortcut = nothing
    if use_in_shortcut
        # conv_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        weight = rand(1, 1, in_channels, out_channels)
        bias = rand(1, 1, out_channels, 1)
        conv_shortcut = torch.Conv2d(weight, bias; stride=1, padding=0)
    end
    
    ResnetBlock2D(pre_norm, in_channels, out_channels, use_conv_shortcut, time_embedding_norm, 
        output_scale_factor, norm1, conv1, time_emb_proj, norm2, dropout, conv2, non_linearity,
        use_in_shortcut, conv_shortcut)
end

function (resnet::ResnetBlock2D)(input_tensor, temb)
    hidden_states = input_tensor
    hidden_states = resnet.norm1(hidden_states)

    # TODO: select activation once, call F.swish like functions
    if occursin("swish", resnet.nonlinearity)
        hidden_states = NNlib.swish(hidden_states)
    elseif occursin("silu", resnet.nonlinearity)
        hidden_states = NNlib.swish(hidden_states)
    else
        error("Not implemented $(resnet.non_linearity)")
    end

    hidden_states = resnet.conv1(hidden_states)

    if temb !== nothing
        if occursin("swish", resnet.nonlinearity)
            temb = NNlib.swish(temb)
        elseif occursin("silu", resnet.nonlinearity)
            temb = NNlib.swish(temb)
        else
            error("Not implemented $(resnet.non_linearity)")
        end
        temb = resnet.time_emb_proj(temb)
        temb = reshape(temb, (1, 1, size(temb)...))
    end


    if temb !== nothing && resnet.time_embedding_norm == "default"
        hidden_states = hidden_states .+ temb
    end

    hidden_states = resnet.norm2(hidden_states)

    if temb !== nothing && resnet.time_embedding_norm == "scale_shift"
        scale, shift = MLUtils.chunk(temb, 2, dims=1)
        hidden_states = hidden_states * (1 + scale) + shift
    end

    if occursin("swish", resnet.nonlinearity)
        hidden_states = NNlib.swish(hidden_states)
    elseif occursin("silu", resnet.nonlinearity)
        hidden_states = NNlib.swish(hidden_states)
    else
        error("Not implemented $(resnet.non_linearity)")
    end

    hidden_states = resnet.dropout(hidden_states)
    hidden_states = resnet.conv2(hidden_states)

    if resnet.conv_shortcut !== nothing
        input_tensor = resnet.conv_shortcut(input_tensor)
    end
    output_tensor = (input_tensor .+ hidden_states) / resnet.output_scale_factor
    return output_tensor
end

function load_state!(attn::ResnetBlock2D, state)
    for k in keys(state)
        key = getfield(attn, k)
        val = getfield(state, k)
        load_state!(key, val)
      end
end
