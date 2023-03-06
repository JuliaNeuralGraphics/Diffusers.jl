struct CrossAttnDownBlock2D
    resnets::torch.ModuleList
    attentions::torch.ModuleList
    downsamplers::Union{Nothing,torch.ModuleList}
end

@functor CrossAttnDownBlock2D

function CrossAttnDownBlock2D(
        in_channels::Integer,
        out_channels::Integer,
        temb_channels::Integer;
        dropout::Float32=0.0f0,
        num_layers::Integer=1,
        resnet_eps::Float64=1^-6,
        resnet_time_scale_shift::String="default",
        resnet_act_fn::String="swish",
        resnet_groups::Integer=32,
        resnet_pre_norm::Bool=true,
        attn_num_head_channels=1,
        cross_attention_dim=1280,
        output_scale_factor=1.0f0,
        downsample_padding=1,
        add_downsample=true,
        dual_cross_attention=false,
        use_linear_projection=false,
        only_cross_attention=false,
        upcast_attention=false,
    )
    resnets = []
    attentions = []
    has_cross_attention = true

    for i in 1:num_layers
        in_channels = i == 1 ? in_channels : out_channels
        push!(resnets, ResnetBlock2D(in_channels;
            out_channels=out_channels,
            temb_channels=temb_channels,
            eps=resnet_eps,
            groups=resnet_groups,
            dropout=dropout,
            time_embedding_norm=resnet_time_scale_shift,
            non_linearity=resnet_act_fn,
            output_scale_factor=output_scale_factor,
            pre_norm=resnet_pre_norm,
        ))

        push!(attentions, Transformer2DModel(; num_attention_heads=attn_num_head_channels,
                attention_head_dim=Int(out_channels / attn_num_head_channels),
                in_channels=out_channels,
                num_layers=1,
                cross_attention_dim=cross_attention_dim,
                norm_num_groups=resnet_groups,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention,
                upcast_attention=upcast_attention,
        ))
    end
    attentions = torch.ModuleList(attentions)
    resnets = torch.ModuleList(resnets)

    if add_downsample
        weight = rand(3, 3, in_channels, out_channels)
        bias = rand(1, 1, out_channels, 1)
        conv = [torch.Conv2d(weight, bias; stride=2, padding=downsample_padding)]
        downsamplers = torch.ModuleList(conv)
    end
    CrossAttnDownBlock2D(resnets, attentions, downsamplers)
end

function (crossattn::CrossAttnDownBlock2D)(hidden_states; 
    temb=nothing, encoder_hidden_states=nothing, 
    attention_mask=nothing, cross_attention_kwargs=nothing)
    # attention_mask is not used
    output_states = []
    for (index, (resnet, attn)) in enumerate(zip(crossattn.resnets, crossattn.attentions))
        hidden_states = resnet(hidden_states, temb)
        hidden_states = attn(hidden_states; encoder_hidden_states=encoder_hidden_states,
            cross_attention_kwargs=cross_attention_kwargs,
        ).sample
    end
    push!(output_states, hidden_states)
    if crossattn.downsamplers !== nothing
        for downsampler in crossattn.downsamplers
            hidden_states = downsampler(hidden_states)
        end
        push!(output_states, hidden_states)
    end
    return hidden_states, output_states
end

function load_state!(layer::CrossAttnDownBlock2D, state)
    for k in keys(state)
        key = getfield(layer, k)
        val = getfield(state, k)
        if k == :downsamplers
            load_state!(key[1], val[1].conv)
        else
            load_state!(key, val)
        end
    end
end