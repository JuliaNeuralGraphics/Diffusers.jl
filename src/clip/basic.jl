struct CLIPTextEmbeddings{T, P, I <: AbstractMatrix{Int32}}
    token_embedding::T
    position_embedding::P
    position_ids::I
end
Flux.@functor CLIPTextEmbeddings

# TODO better way to handle fixed int type during f16 conversion
function CLIPTextEmbeddings(token_embedding, position_embedding, position_ids)
    pi = eltype(position_ids) == Int32 ? position_ids : Int32.(position_ids)
    CLIPTextEmbeddings(token_embedding, position_embedding, pi)
end

Flux.trainable(emb::CLIPTextEmbeddings) = (emb.token_embedding, emb.position_embedding)

function CLIPTextEmbeddings(;
    vocab_size::Int, embed_dim::Int, max_position_embeddings::Int,
)
    CLIPTextEmbeddings(
        Embedding(vocab_size => embed_dim),
        Embedding(max_position_embeddings => embed_dim),
        reshape(collect(UnitRange{Int32}(1, max_position_embeddings)), :, 1))
end

function (emb::CLIPTextEmbeddings)(input_ids::I) where {
    I <: AbstractMatrix{Int32}
}
    seq_length = size(input_ids, 1)
    position_ids = emb.position_ids[1:seq_length, :]
    emb.token_embedding(input_ids) .+ emb.position_embedding(position_ids)
end
