struct CLIPTokenizer
    encoder::Dict{String, Int64}
    decoder::Dict{Int64, String}
    byte_encoder::Dict{UInt64, Char}
    byte_decoder::Dict{Char, UInt64}
    vocab::Vector{String}
    bpe_ranks::Dict{NTuple{2, String}, Int64}
    cache::Dict{String, String}
    pattern::Regex
end

function CLIPTokenizer(; bpe_path::String = joinpath(pkgdir(Diffusers), "data", "bpe_simple_vocab_16e6.txt"))
    merges = [
        tuple(split(merge)...)
        for merge in split(read(bpe_path, String), '\n')[2:48895]]

    byte_encoder = bytes_to_unicode()
    byte_decoder = Dict{Char, UInt64}(v => k for (k, v) in byte_encoder)

    vocab = string.(values(byte_encoder))
    append!(vocab, v * "</w>" for v in vocab) # `</w>` denotes whitespace.
    append!(vocab, join(merge) for merge in merges)
    append!(vocab, ("<|startoftext|>", "<|endoftext|>"))
    @assert length(vocab) == 49408

    encoder = Dict{String, Int64}(zip(vocab, 1:length(vocab)))
    decoder = Dict{Int64, String}(v => k for (k, v) in encoder)
    bpe_ranks = Dict{NTuple{2, String}, Int64}(zip(merges, 1:length(merges)))
    cache = Dict{String, String}(
        "<|startoftext|>" => "<|startoftext|>",
        "<|endoftext|>" => "<|endoftext|>")

    pattern = r"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+"
    CLIPTokenizer(
        encoder, decoder, byte_encoder, byte_decoder,
        vocab, bpe_ranks, cache, pattern)
end

n_tokens(tk::CLIPTokenizer) = length(tk.vocab)

function encode(tk::CLIPTokenizer, text::String)
    bpe_tokens = Int64[]
    for match in eachmatch(tk.pattern, clean_text(text))
        token = join(tk.byte_encoder[convert(UInt64, c)] for c in match.match)
        bpe_token = bpe(tk, token)
        append!(bpe_tokens, tk.encoder[bt] for bt in split(bpe_token))
    end
    bpe_tokens
end

# TODO longet text context length
# TODO encode in Int32
# TODO encode in 1-based indexing
# TODO
#   what is initialization value? can't be 0, because embedding fails
#   probably does not matter since it is masked away
#   but for the sake of it, probably should be <|endoftext|>
function tokenize(
    tk::CLIPTokenizer, texts::Vector{String};
    context_length::Int, truncate::Bool = false,
    add_start_end::Bool = false,
)
    n = length(texts)
    encodings = [
        encode(tk, add_start_end ? "<|startoftext|> $text <|endoftext|>" : text)
        for text in texts]

    # tokens = zeros(Int64, context_length, n)
    tokens = fill(1, context_length, n)
    pad_mask = fill(false, context_length, n)
    for (i, enc) in enumerate(encodings)
        if length(enc) > context_length
            truncate || error("""
                [$i/$n] encoded input text is too long for context length `$context_length`.
                Either set `truncate=true` or increase the `context_length`.
                Text: `$(texts[i])`.
                Encoding: `$enc`.
            """)
            enc = enc[1:context_length]
        end
        tokens[1:length(enc), i] .= enc
        pad_mask[1:length(enc), i] .= true
    end
    tokens, pad_mask
end

function decode(
    tk::CLIPTokenizer, tokens::T;
    remove_start_end::Bool = true, ignore_padding::Bool = true,
) where T <: AbstractVector{Int64} # TODO Int32
    if remove_start_end
        eof_tokens = (tk.encoder["<|startoftext|>"], tk.encoder["<|endoftext|>"])
        tokens = [t for t in tokens if !(t in eof_tokens)]
    end
    if ignore_padding
        pad_idx = findfirst(t -> t == 0, tokens)
        isnothing(pad_idx) || (tokens = tokens[1:pad_idx - 1];)
    end

    raw_text = join([tk.decoder[t] for t in tokens])
    replace(String(UInt8[tk.byte_decoder[c] for c in raw_text]), "</w>" => " ")
end

function bpe(tk::CLIPTokenizer, token::String)
    token in keys(tk.cache) && return tk.cache[token]

    word = push!([string(t) for t in token[1:end - 1]], "$(token[end])</w>")
    pairs = get_pairs(word)
    isempty(pairs) && return "$token</w>"

    while true
        bigram_rank, bigram_idx = findmin(pair -> get(tk.bpe_ranks, pair, Inf), pairs)
        bigram_rank == Inf && break

        bigram = pairs[bigram_idx]
        b1, b2 = bigram
        new_word = String[]
        i = 1
        while i ≤ length(word)
            j = findfirst(w -> w == b1, @view(word[i:end]))
            if isnothing(j)
                append!(new_word, @view(word[i:end]))
                break
            else
                j += i - 1 # Offset to represent index in `word`, not its view.
                append!(new_word, @view(word[i:(j - 1)]))
                i = j
            end

            if (word[i] == b1) && (i < length(word)) && (word[i + 1] == b2)
                push!(new_word, b1 * b2)
                i += 2
            else
                push!(new_word, word[i])
                i += 1
            end
        end

        word = copy(new_word)
        length(word) == 1 && break
        pairs = get_pairs(word)
    end

    bpe_word = join(word, " ")
    tk.cache[token] = bpe_word
    bpe_word
end

function get_pairs(word::Vector{String})
    pairs = OrderedSet{NTuple{2, String}}()
    prev_char = word[1]
    for char in @view(word[2:end])
        push!(pairs, (prev_char, char))
        prev_char = char
    end
    pairs
end

function bytes_to_unicode()
    bytes = append!(
        collect(convert(UInt64, '!'):convert(UInt64, '~')),
        collect(convert(UInt64, '¡'):convert(UInt64, '¬')),
        collect(convert(UInt64, '®'):convert(UInt64, 'ÿ')))
    chars = deepcopy(bytes)

    char_offset::UInt64 = 2^8 # Avoid mapping to whitespaces, which BPE does not like.
    n::UInt64 = 0
    for b in 0:(2^8 - 1)
        b in bytes && continue
        push!(bytes, b)
        push!(chars, n + char_offset)
        n += 1
    end

    OrderedDict{UInt64, Char}(zip(bytes, map(c -> convert(Char, c), chars)))
end

function clean_text(text::String)
    text = lowercase(text)
    text = strip(text)
    text = replace(text, r"\s+" => " ")
    strip(text)
end
