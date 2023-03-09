struct CrossAttnDownBlock2D{R, A}
    resnets::R
    attentions::A
    downsamplers::Maybe{Chain}
end

Flux.@functor CrossAttnDownBlock2D

function CrossAttnDownBlock2D(;
        in_channels::Int, out_channels::Int, time_emb_channels::Int,
        dropout::Real=0, n_layers::Int=1, resnet_time_scale_shift::Bool=false, 
        resnet_λ=swish, resnet_groups::Integer=32, attn_n_heads=1, down_padding=1, 
        context_dim=1280, add_downsample=true, use_linear_projection=false)
    resnets = []
    attentions = []
    
    resnets = Chain([ResnetBlock2D(;in_channels=(i == 1 ? in_channels : out_channels), 
        λ = resnet_λ, out_channels=out_channels, time_emb_channels=time_emb_channels, 
        n_groups=resnet_groups, dropout=dropout, 
        embedding_scale_shift=resnet_time_scale_shift) for i in 1:n_layers]...)
    
    attentions = Chain([Transformer2D(; in_channels=out_channels, n_heads=attn_n_heads, 
        dropout=dropout, head_dim=Int(out_channels/attn_n_heads), 
        n_norm_groups=resnet_groups, use_linear_projection=use_linear_projection, 
        context_dim=context_dim,) for i in 1:n_layers]...)

    downsamplers = nothing
    if add_downsample
        downsamplers = Chain(
            Conv((3, 3), in_channels => out_channels; stride=2, pad=down_padding))
    end
    CrossAttnDownBlock2D(resnets, attentions, downsamplers)
end


function (cattn::CrossAttnDownBlock2D)(
    x::T, time_emb::Maybe{E} = nothing, context::Maybe{C} = nothing) where { 
        T <: AbstractArray{Float32, 4},
        E <: AbstractArray{Float32, 2},
        C <: AbstractArray{Float32, 3},
}
    # Note: attention_mask is not used
    for (index, (resnet, attn)) in enumerate(zip(cattn.resnets, cattn.attentions))
        x = resnet(x, time_emb)
        x = attn(x, context)
    end

    if cattn.downsamplers ≢ nothing
        x = cattn.downsamplers(x)
    end
    return x
end