struct CrossAttnDownBlock2D{R, A, D}
    resnets::R
    attentions::A
    downsamplers::D
end

Flux.@functor CrossAttnDownBlock2D

function CrossAttnDownBlock2D(;
        in_channels::Int, out_channels::Int, time_emb_channels::Int,
        dropout::Real = 0, n_layers::Int = 1, resnet_time_scale_shift::Bool = false, 
        resnet_λ = swish, resnet_groups::Int = 32, attn_n_heads::Int = 1, 
        down_padding::Int = 1, context_dim::Int = 1280, add_downsample::Bool = true, 
        use_linear_projection::Bool = false)

    resnets = Chain([ResnetBlock2D(;in_channels=(i == 1 ? in_channels : out_channels), 
        λ = resnet_λ, out_channels, time_emb_channels, n_groups=resnet_groups, dropout, 
        embedding_scale_shift=resnet_time_scale_shift) for i in 1:n_layers]...)
    
    attentions = Chain([Transformer2D(; in_channels=out_channels, n_heads=attn_n_heads, 
        dropout, head_dim=out_channels÷attn_n_heads, n_norm_groups=resnet_groups, 
        use_linear_projection, context_dim) for i in 1:n_layers]...)

    downsamplers = isnothing(add_downsample) ? identity : Chain(
        Conv((3, 3), in_channels => out_channels; stride=2, pad=down_padding))

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
    x = cattn.downsamplers(x)
    return x
end

# UNetMidBlock2DCrossAttn in diffusers.py
struct CrossAttnMidBlock2D{R, A}
    resnets::R
    attentions::A
end

Flux.@functor CrossAttnMidBlock2D

function CrossAttnMidBlock2D(;
    in_channels::Int, time_emb_channels::Int, dropout::Real = 0, n_layers::Int=1,
    resnet_time_scale_shift::Bool = false, resnet_λ = swish, resnet_groups::Int=32,
    attn_n_heads::Int=1, context_dim::Int=1280, use_linear_projection::Bool=false)

    resnet_groups = isnothing(resnet_groups) ? min(in_channels÷4, 32) : resnet_groups

    resnets = [ ResnetBlock2D(;
        in_channels, out_channels=in_channels, time_emb_channels, dropout, λ = resnet_λ, 
        n_groups=resnet_groups, embedding_scale_shift=resnet_time_scale_shift)]
    attentions = []

    for i in 1:n_layers
        push!(attentions, Transformer2D(; in_channels, n_heads=attn_n_heads, context_dim, 
            dropout, use_linear_projection, head_dim=in_channels÷attn_n_heads, 
            n_norm_groups=resnet_groups))
        push!(resnets, ResnetBlock2D(; in_channels, out_channels=in_channels, dropout, 
            λ = resnet_λ, time_emb_channels, n_groups=resnet_groups, 
            embedding_scale_shift=resnet_time_scale_shift))
    end

    CrossAttnMidBlock2D(Chain(resnets...), Chain(attentions...))
end

function (mid::CrossAttnMidBlock2D)(
    x::T, time_emb::Maybe{E} = nothing, context::Maybe{C} = nothing) where { 
        T <: AbstractArray{Float32, 4},
        E <: AbstractArray{Float32, 2},
        C <: AbstractArray{Float32, 3},
}
    x = mid.resnets[1](x, time_emb)
    for (index, (resnet, attn)) in enumerate(zip(mid.resnets[2:end], mid.attentions))
        x = attn(x, context)
        x = resnet(x, time_emb)
    end
    x
end
