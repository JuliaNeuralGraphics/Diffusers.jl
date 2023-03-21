quick_gelu(x) = x * sigmoid(1.702f0 * x)

struct CLIPMLP{F1, F2}
    fc1::F1
    fc2::F2
end
Flux.@functor CLIPMLP

function CLIPMLP(dims::Pair{Int, Int}, λ = quick_gelu)
    CLIPMLP(Dense(dims, λ), Dense(reverse(dims)))
end

function (mlp::CLIPMLP)(x::T) where T <: AbstractArray{Float32, 3}
    mlp.fc2(mlp.fc1(x))
end

struct CLIPAttention{T, D}
    q::T
    k::T
    v::T
    out::T
    dropout::D

    n_heads::Int
    head_dim::Int
end
Flux.@functor CLIPAttention

function CLIPAttention(dim::Int; n_heads::Int, dropout::Real = 0)
    head_dim = dim ÷ n_heads
    q = Dense(dim => dim)
    k = Dense(dim => dim)
    v = Dense(dim => dim)
    out = Dense(dim => dim)
    CLIPAttention(
        q, k, v, out,
        iszero(dropout) ? identity : Dropout(dropout),
        n_heads, head_dim)
end

function (attn::CLIPAttention)(
    x::T; mask::Maybe{M1} = nothing, causal_mask::Maybe{M2} = nothing,
) where {
    T <: AbstractArray{Float32, 3},
    M1 <: AbstractMatrix{Bool},
    M2 <: AbstractMatrix{Bool},
}
    _, target_length, batch = size(x)
    q, k, v = attn.q(x), attn.k(x), attn.v(x)

    # Combine attention mask & causal mask.
    # Reshape masks to `(kv len, q len, n heads, batch)` shape.
    mask = isnothing(mask) ? mask :
        reshape(mask, (size(mask, 1), 1, 1, size(mask, 2)))
    if !isnothing(causal_mask)
        causal_mask = reshape(causal_mask, size(causal_mask)..., 1, 1)
        mask = isnothing(mask) ? causal_mask : (causal_mask .| mask)
    end

    ω, _ = dot_product_attention(
        q, k, v; mask, nheads=attn.n_heads, fdrop=attn.dropout)
    attn.out(reshape(ω, :, target_length, batch))
end

struct CLIPEncoderLayer{A, L1, M, L2}
    self_attn::A
    layer_norm1::L1
    layer_norm2::L2
    mlp::M
end
Flux.@functor CLIPEncoderLayer

function CLIPEncoderLayer(
    dim::Int; intermediate_size::Int, n_heads::Int, dropout::Real = 0,
    λ = quick_gelu,
)
    self_attn = CLIPAttention(dim; n_heads, dropout)
    layer_norm1 = LayerNorm(dim)
    layer_norm2 = LayerNorm(dim)
    mlp = CLIPMLP(dim => intermediate_size, λ)
    CLIPEncoderLayer(self_attn, layer_norm1, layer_norm2, mlp)
end

function (enc::CLIPEncoderLayer)(
    x::T; mask::Maybe{M1} = nothing, causal_mask::Maybe{M2} = nothing,
) where {
    T <: AbstractArray{Float32, 3},
    M1 <: AbstractMatrix{Bool},
    M2 <: AbstractMatrix{Bool},
}
    residual = x
    x = enc.layer_norm1(x)
    x = enc.self_attn(x; mask, causal_mask)
    x = residual .+ x

    residual = x
    x = enc.layer_norm2(x)
    x = enc.mlp(x)
    residual .+ x
end

struct CLIPEncoder{L}
    layers::L
end
Flux.@functor CLIPEncoder

function CLIPEncoder(
    dim::Int; intermediate_size::Int, n_heads::Int, num_hidden_layers::Int,
    dropout::Real = 0, λ = quick_gelu,
)
    CLIPEncoder(Chain([
        CLIPEncoderLayer(dim; intermediate_size, n_heads, dropout, λ)
        for i in 1:num_hidden_layers]...))
end

function (enc::CLIPEncoder)(
    x::T; mask::Maybe{M1} = nothing, causal_mask::Maybe{M2} = nothing
) where {
    T <: AbstractArray{Float32, 3},
    M1 <: AbstractMatrix{Bool},
    M2 <: AbstractMatrix{Bool},
}
    for layer in enc.layers
        x = layer(x; mask, causal_mask)
    end
    x
end

struct CLIPTextTransformer{B, E, L}
    embeddings::B
    encoder::E
    final_layer_norm::L
end
Flux.@functor CLIPTextTransformer

function CLIPTextTransformer(;
    vocab_size::Int, embed_dim::Int, max_position_embeddings::Int,
    n_heads::Int, num_hidden_layers::Int, intermediate_size::Int,
    dropout::Real = 0, λ = quick_gelu,
)
    embeddings = CLIPTextEmbeddings(; vocab_size, embed_dim, max_position_embeddings)
    encoder = CLIPEncoder(
        embed_dim; intermediate_size, n_heads,
        num_hidden_layers, dropout, λ)
    final_layer_norm = LayerNorm(embed_dim)
    CLIPTextTransformer(embeddings, encoder, final_layer_norm)
end

function (transformer::CLIPTextTransformer)(
    input_ids::I; mask::Maybe{M} = nothing,
) where {
    I <: AbstractMatrix{Int32},
    M <: AbstractMatrix{Bool},
}
    x = transformer.embeddings(input_ids)
    causal_mask = make_causal_mask(input_ids; dims=1)
    x = transformer.encoder(x; mask, causal_mask)
    transformer.final_layer_norm(x)
end

# HGF integration.

function CLIPTextTransformer(model_name::String; state_file::String, config_file::String)
    state, cfg = load_pretrained_model(model_name; state_file, config_file)
    transformer = CLIPTextTransformer(;
        vocab_size=cfg["vocab_size"],
        embed_dim=cfg["hidden_size"],
        max_position_embeddings=cfg["max_position_embeddings"],

        n_heads=cfg["num_attention_heads"],
        num_hidden_layers=cfg["num_hidden_layers"],
        intermediate_size=cfg["intermediate_size"],
        dropout=cfg["dropout"])
    load_state!(transformer, state.text_model)
    transformer
end
