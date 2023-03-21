struct Embedding{E}
    weights::E
end
Flux.@functor Embedding

function Embedding(; embed_dim::Int, vocab_size::Int)
    Embedding(randn(Float32, embed_dim, vocab_size))
end

# ids are 1-based
function (e::Embedding)(ids::T) where T <: AbstractMatrix{<: Integer}
    NNlib.gather(e.weights, ids)
end

struct CLIPTextEmbeddings{T, P, I}
    token_embedding::T
    position_embedding::P
    position_ids::I
end
Flux.@functor CLIPTextEmbeddings

Flux.trainable(emb::CLIPTextEmbeddings) = (emb.token_embedding, emb.position_embedding)

function CLIPTextEmbeddings(;
    vocab_size::Int, embed_dim::Int, max_position_embeddings::Int,
)
    CLIPTextEmbeddings(
        Embedding(; embed_dim, vocab_size),
        Embedding(; embed_dim, vocab_size=max_position_embeddings),
        reshape(collect(UnitRange{Int32}(1, max_position_embeddings)), :, 1))
end

function (emb::CLIPTextEmbeddings)(input_ids::I) where {
    I <: AbstractMatrix{Int32}
}
    seq_length = size(input_ids, 1)
    position_ids = emb.position_ids[1:seq_length, :]
    emb.token_embedding(input_ids) .+ emb.position_embedding(position_ids)
end
