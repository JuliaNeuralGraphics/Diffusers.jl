struct UNet2DCondition{CI, S, T, D, M, U, G, CO}
    conv_in::CI
    sin_embedding::S
    time_embedding::T

    down_blocks::D
    mid_block::M
    up_blocks::U

    conv_norm_out::G
    conv_out::CO
end

function UNet2DCondition(
    channels::Pair{Int, Int} = 4 => 4;
    n_layers::Int = 2,
    n_groups::Int = 32,
    embedding_scale_shift::Bool = false,
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
    downsample_padding::Int = 1,
)
    in_channels, out_channels = channels
    conv_in = Conv((3, 3), in_channels => block_out_channels[1]; pad=(1, 1))

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
                input_channel => output_channel,
                time_emb_channels; n_layers, n_groups, embedding_scale_shift,
                pad=downsample_padding, add_downsample=!is_last)
        elseif down_block <: CrossAttnDownBlock2D
            CrossAttnDownBlock2D(
                input_channel => output_channel;
                time_emb_channels, n_layers, embedding_scale_shift,
                n_groups, n_heads, context_dim,
                pad=downsample_padding, add_downsample=!is_last)
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
                add_upsample=!is_last)
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

    UNet2DCondition(
        conv_in, sin_embedding, time_embedding,
        (down_blocks...,), mid_block, Chain(up_blocks...),
        conv_norm_out, conv_out)
end

function (unet::UNet2DCondition)(
    x::X, timestep::T, text_emb::C
) where {
   X <: AbstractArray{Float32, 4},
   T <: AbstractVector{Int32},
   C <: AbstractArray{Float32, 3}
}
    time_emb = unet.sin_embedding(timestep)
    time_emb = unet.time_embedding(time_emb)

    function _chain(blocks::Tuple, h)
        block = first(blocks)
        h, states = typeof(block) <: DownBlock2D ?
            block(h, time_emb) :
            block(h, time_emb, text_emb)
        (states..., _chain(Base.tail(blocks), h)...)
    end
    _chain(::Tuple{}, _) = ()

    x = unet.conv_in(x)
    states = (x, _chain(unet.down_blocks, x)...)
    states, x = states[end:-1:1], states[end]

    x = unet.mid_block(x, time_emb, text_emb)

    for block in unet.up_blocks
        x, states = typeof(block) <: UpBlock2D ?
            block(x, states, time_emb) :
            block(x, states, time_emb, text_emb)
    end
    unet.conv_out(unet.conv_norm_out(x))
end

# HGF integration.

function UNet2DCondition(model_name::String; state_file::String, config_file::String)
    state, cfg = load_pretrained_model(model_name; state_file, config_file)
    unet = UNet2DCondition(
        cfg["in_channels"] => cfg["out_channels"];
        n_heads=cfg["attention_head_dim"],
        freq_shift=cfg["freq_shift"],
        n_layers=cfg["layers_per_block"],
        n_groups=cfg["norm_num_groups"],
        context_dim=cfg["cross_attention_dim"],
        block_out_channels=(cfg["block_out_channels"]...,),
        downsample_padding=cfg["downsample_padding"],
    )
    load_state!(unet, state)
    unet
end
