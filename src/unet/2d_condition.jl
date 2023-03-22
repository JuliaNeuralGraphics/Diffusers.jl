struct UNet2DConditionModel{CI <: Conv, S, T, D, M, U, G <: GroupNorm, CO <: Conv}
    conv_in::CI
    sin_embedding::S
    time_embedding::T
    down_blocks::D
    mid_block::M
    up_blocks::U
    conv_norm_out::G
    conv_out::CO
end

function UNet2DConditionModel(;
    in_channels::Int = 4,
    out_channels::Int = 4,
    n_layers::Int = 2,
    n_groups::Int = 32,
    resnet_time_scale_shift::Bool = false,
    context_dim::Int = 1280,
    freq_shift::Int = 0,
    n_heads::Int = 8,
    down_block_types = (
        CrossAttnDownBlock2D,
        CrossAttnDownBlock2D,
        CrossAttnDownBlock2D,
        DownBlock2D,
    ),
    up_block_types = (
        UpBlock2D,
        CrossAttnUpBlock2D,
        CrossAttnUpBlock2D,
        CrossAttnUpBlock2D
    ),
    block_out_channels=(320, 640, 1280, 1280),
)
    # preprocess
    conv_in = Conv((3, 3), in_channels => block_out_channels[1]; pad=(1, 1))

    # time
    time_emb_channels = block_out_channels[1] * 4
    sin_embedding = SinusoidalEmbedding(block_out_channels[1]; freq_shift)
    time_embedding = TimestepEmbedding(block_out_channels[1];
        time_embed_dim=time_emb_channels)
    # NOTE no class embedding

    # down
    output_channel = block_out_channels[1]
    down_blocks = []
    for (i, (down_block, block_out)) in enumerate(zip(down_block_types, block_out_channels))
        is_last = i == length(down_block_types)
        input_channel, output_channel = output_channel, block_out

        down_block = if down_block <: DownBlock2D
            DownBlock2D(
                input_channel => output_channel, time_emb_channels;
                n_layers, n_groups, embedding_scale_shift=resnet_time_scale_shift,
                add_sampler=!is_last)
        elseif down_block <: CrossAttnDownBlock2D
            CrossAttnDownBlock2D(;
                in_channels=input_channel, out_channels=output_channel,
                time_emb_channels, n_layers, resnet_time_scale_shift,
                resnet_groups=n_groups, attn_n_heads=n_heads, context_dim,
                add_downsample=!is_last)
        end
        push!(down_blocks, down_block)
    end

    # mid
    mid_block = CrossAttnMidBlock2D(;
        in_channels=block_out_channels[end],
        time_emb_channels, n_heads, context_dim)

    # up
    up_blocks = []
    block_out_channels = reverse(block_out_channels)
    for (i, (up_block, block_out)) in enumerate(zip(up_block_types, block_out_channels))
        is_last = i == length(up_block_types)

        prev_output_channel = output_channel
        in_idx = min(i + 1, length(block_out_channels))
        input_channel, output_channel = block_out_channels[in_idx], block_out

        up = if up_block <: UpBlock2D
            UpBlock2D(
                input_channel => output_channel, prev_output_channel,
                time_emb_channels; n_layers=n_layers + 1, n_groups,
                add_sampler=!is_last)
        else
            CrossAttnUpBlock2D(
                input_channel => output_channel, prev_output_channel,
                time_emb_channels; attn_n_heads=n_heads, n_layers=n_layers + 1,
                n_groups, context_dim, add_upsample=!is_last)
        end
        push!(up_blocks, up)
    end

    # postprocess
    conv_norm_out = GroupNorm(block_out_channels[end], n_groups, swish)
    conv_out = Conv((3, 3), block_out_channels[end] => out_channels; pad=1)

    UNet2DConditionModel(
        conv_in, sin_embedding, time_embedding,
        Chain(down_blocks...), mid_block, Chain(up_blocks...),
        conv_norm_out, conv_out)
end

function (unet::UNet2DConditionModel)(
    x::X, timestep::T, text_emb::C
) where {
   X <: AbstractArray{Float32, 4},
   T <: AbstractVector{Int32},
   C <: AbstractArray{Float32, 3}
}
    time_emb = unet.sin_embedding(timestep)
    time_emb = unet.time_embedding(time_emb)

    x = unet.conv_in(x)
    skips = [x, ]
    for downblock in unet.down_blocks
        if typeof(downblock) <: DownBlock2D
            x, skip... = downblock(x, time_emb)
        else
            x, skip... = downblock(x, time_emb, text_emb)
        end
        push!(skips, skip...)
    end
    x = unet.mid_block(x, time_emb, text_emb)

    # avoid last skips element & create a tuple with 3 element chunks, all reversed
    skips = reverse(skips[1:end-1])
    skips_chunks = [ tuple(reverse(skips[i:i+2])...) for i in 1:3:length(skips)]

    for (i, upblock) in enumerate(unet.up_blocks)
        res_samples = skips_chunks[i]
        if typeof(upblock) <: UpBlock2D
            x = upblock(x, res_samples, time_emb)
        else
            x = upblock(x, res_samples, time_emb, text_emb)
        end
    end
    unet.conv_out(unet.conv_norm_out(x))
end
