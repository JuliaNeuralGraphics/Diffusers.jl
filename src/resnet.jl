struct ResnetBlock2D{I, O, N, S, E}
    init_proj::I
    out_proj::O
    norm::N
    conv_shortcut::S
    time_emb_proj::E

    embedding_scale_shift::Bool
end
Flux.@functor ResnetBlock2D

function ResnetBlock2D(;
    in_channels::Int, out_channels::Maybe{Int} = nothing,
    n_groups::Int = 32, n_groups_out::Maybe{Int} = nothing,
    embedding_scale_shift::Bool = false, time_emb_channels::Int = 512,
    use_shortcut::Maybe{Bool} = nothing, conv_out_channels::Maybe{Int} = nothing,
    dropout::Real = 0, λ = swish,
)
    out_channels = isnothing(out_channels) ? in_channels : out_channels
    n_groups_out = isnothing(n_groups_out) ? n_groups : n_groups_out
    conv_out_channels = isnothing(conv_out_channels) ? out_channels : conv_out_channels
    use_shortcut = isnothing(use_shortcut) ?
        (in_channels != conv_out_channels) : use_shortcut
    time_emb_out_channels = embedding_scale_shift ?
        (out_channels * 2) : out_channels

    # NOTE no up/down
    init_proj = Chain(
        GroupNorm(in_channels, n_groups, λ),
        Conv((3, 3), in_channels => out_channels; pad=1))
    out_proj = Chain(
        x -> λ.(x),
        iszero(dropout) ? identity : Dropout(dropout),
        Conv((3, 3), out_channels => conv_out_channels; pad=1))

    norm = GroupNorm(out_channels, n_groups_out)
    time_emb_proj = Chain(
        x -> λ.(x),
        Dense(time_emb_channels => time_emb_out_channels))
    conv_shortcut = use_shortcut ?
        Conv((1, 1), in_channels => conv_out_channels) : identity

    ResnetBlock2D(
        init_proj, out_proj, norm, conv_shortcut, time_emb_proj,
        embedding_scale_shift)
end

function (block::ResnetBlock2D)(x::T, time_embedding::Maybe{E}) where {
    T <: AbstractArray{Float32, 4}, E <: AbstractMatrix{Float32},
}
    skip, x = x, block.init_proj(x)

    if time_embedding ≢ nothing
        time_embedding = block.time_emb_proj(time_embedding)
        time_embedding = reshape(time_embedding, 1, 1, size(time_embedding)...)
        block.embedding_scale_shift || (x = x .+ time_embedding;)
    end

    x = block.norm(x)

    if time_embedding ≢ nothing && block.embedding_scale_shift
        scale, shift = chunk(time_embedding, 2; dims=3)
        x = x .* (1f0 .+ scale) .+ shift
    end

    block.out_proj(x) .+ block.conv_shortcut(skip)
end
