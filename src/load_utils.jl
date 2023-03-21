"""
    load_pretrained_model(model_name::String, config::String, bin_file::String)

Loads config and model state dict from HuggingFace library.
For example:
```julia
julia> state_dict, cfg = load_pretrained_model("runwayml/stable-diffusion-v1-5", "unet/config.json", "unet/diffusion_pytorch_model.bin")
```
"""
function load_pretrained_model(
    model_name::String; state_file::String, config_file::String,
)
    config = load_hgf_config(model_name; filename=config_file)
    state_url = HuggingFaceURL(model_name, state_file)
    state = Pickle.Torch.THload(_hgf_download(state_url))
    state_dict_to_namedtuple(state), config
end

function load_hgf_config(model_name::String; filename::String)
    url = HuggingFaceURL(model_name, filename)
    JSON3.read(read(_hgf_download(url)))
end

function _hgf_download(
    url::HuggingFaceURL; cache::Bool = true,
    auth_token = HuggingFaceApi.get_token(),
)
    hf_hub_download(
        url.repo_id, url.filename; repo_type=url.repo_type,
        revision=url.revision, auth_token, cache)
end

function state_dict_to_namedtuple(state_dict)
    ht = Pickle.HierarchicalTable()
    foreach(((k, v),) -> setindex!(ht, v, k), pairs(state_dict))
    _ht2nt(ht)
end

_ht2nt(x::Some) = something(x)
_ht2nt(x::Pickle.HierarchicalTable) = _ht2nt(x.head)
function _ht2nt(x::Pickle.TableBlock)
    if iszero(length(x.entry))
        return ()
    else
        tks = Tuple(keys(x.entry))
        if all(Base.Fix1(all, isdigit), tks)
            n_indices = maximum(parse.(Int, tks)) + 1
            inds = Vector(undef, n_indices)
            foreach(tks) do is
                i = parse(Int, is) + 1
                inds[i] = _ht2nt(x.entry[is])
            end
            return inds
        else
            cs = map(_ht2nt, values(x.entry))
            ns = map(Symbol, tks)
            return NamedTuple{ns}(cs)
        end
    end
end

# TODO verify keys before loop and fail early

function load_state!(layer::Flux.Conv, state)
    for k in keys(state)
        v = getfield(state, k)
        if k == :weight
            # BCHW -> WHCB & flip kernel from cross-correlation to convolution.
            v = permutedims(v, (4, 3, 2, 1))[end:-1:1, end:-1:1, :, :]
        end
        getfield(layer, k) .= v
    end
end

function load_state!(layer::Flux.Dense, state)
    for k in keys(state)
        getfield(layer, k) .= getfield(state, k)
    end
end

function load_state!(chain::Flux.Chain, state)
    for (i, layer) in enumerate(chain)
        (layer isa Dropout || layer ≡ identity) && continue
        load_state!(layer, state[i])
    end
end

function load_state!(layer::Flux.LayerNorm, state)
    for k in keys(state)
        key = getfield(layer.diag, k == :weight ? :scale : k)
        val = getfield(state, k)
        key .= val
    end
end

function load_state!(layer::Flux.GroupNorm, state)
    layer.γ .= state.weight
    layer.β .= state.bias
    return nothing
end

function load_state!(attn::Attention, state; use_cross_attention::Bool = false)
    if cross_attention(attn) || use_cross_attention
        load_state!(getfield(attn, :to_q), getfield(state, :to_q))
        load_state!(getfield(attn, :to_k), getfield(state, :to_k))
        load_state!(getfield(attn, :to_v), getfield(state, :to_v))
        load_state!(getfield(attn, :to_out), getfield(state, :to_out))

        :norm_cross in keys(state) &&
            load_state!(getfield(attn, :norm), getfield(state, :norm_cross))
    else
        load_state!(getfield(attn, :to_q), getfield(state, :query))
        load_state!(getfield(attn, :to_k), getfield(state, :key))
        load_state!(getfield(attn, :to_v), getfield(state, :value))
        load_state!(getfield(attn, :to_out)[1], getfield(state, :proj_attn))
        load_state!(getfield(attn, :norm), getfield(state, :group_norm))
    end
end

function load_state!(fwd::FeedForward, state)
    load_state!(fwd.fn[1], state.net[1].proj)
    load_state!(fwd.fn[4], state.net[3])
end

function load_state!(block::TransformerBlock, state)
    load_state!(block.attention_1, state.attn1)
    (:attn2) in keys(state) && load_state!(block.attention_2, state.attn2)

    load_state!(block.fwd, state.ff)
    load_state!(block.norm_1, state.norm1)
    load_state!(block.norm_2, state.norm2)
    load_state!(block.norm_3, state.norm3)
end

function load_state!(tr::Transformer2D, state)
    for k in keys(state)
        load_state!(getfield(tr, k), getfield(state, k))
    end
end

function load_state!(block::ResnetBlock2D, state)
    load_state!(block.init_proj[1], state.norm1)
    load_state!(block.init_proj[2], state.conv1)
    load_state!(block.out_proj[3], state.conv2)
    load_state!(block.norm, state.norm2)

    :time_emb_proj in keys(state) && load_state!(
        block.time_emb_proj[2], state.time_emb_proj)

    (block.conv_shortcut ≡ identity) || load_state!(
        block.conv_shortcut, state.conv_shortcut)
end

function load_state!(tr::CrossAttnDownBlock2D, state)
    for k in keys(state)
        if k == :downsamplers
            load_state!(getfield(tr, k)[1], getfield(state, k)[1].conv) # inside .conv
        else
            load_state!(getfield(tr, k), getfield(state, k)) 
        end
    end
end

function load_state!(tr::CrossAttnMidBlock2D, state)
    for k in keys(state)
        load_state!(getfield(tr, k), getfield(state, k))
    end
end

function load_state!(vae::AutoencoderKL, state)
    for k in keys(state)
        load_state!(getfield(vae, k), getfield(state, k))
    end
end

function load_state!(enc::Encoder, state)
    load_state!(enc.conv_in, state.conv_in)
    load_state!(enc.conv_out, state.conv_out)
    load_state!(enc.norm, state.conv_norm_out)
    load_state!(enc.mid_block, state.mid_block)
    load_state!(enc.down_blocks, state.down_blocks)
end

function load_state!(dec::Decoder, state)
    load_state!(dec.conv_in, state.conv_in)
    load_state!(dec.conv_out, state.conv_out)
    load_state!(dec.norm, state.conv_norm_out)
    load_state!(dec.mid_block, state.mid_block)
    load_state!(dec.up_blocks, state.up_blocks)
end

function load_state!(mb::MidBlock2D, state)
    for k in keys(state)
        load_state!(getfield(mb, k), getfield(state, k))
    end
end

function load_state!(sampler::SamplerBlock2D{R, S}, state) where {R, S}
    load_state!(sampler.resnets, state.resnets)
    !(:downsamplers in keys(state) || :upsamplers in keys(state)) && return

    sampler_key = S <: Downsample2D ? (:downsamplers) : (:upsamplers)
    load_state!(sampler.sampler, getfield(state, sampler_key)[1])
end

function load_state!(down::Downsample2D, state)
    load_state!(down.conv, state.conv)
end

load_state!(up::Upsample2D, state) = load_state!(up.conv, state.conv)

function load_state!(u::CrossAttnUpBlock2D, state)
    load_state!(u.resnets, state.resnets)
    load_state!(u.attentions, state.attentions)
    if typeof(u.sampler) <: Upsample2D
        load_state!(u.sampler.conv, state.upsamplers[1].conv)
    end
end

function load_state!(u::UpBlock2D, state)
    load_state!(u.resnets, state.resnets)
    if typeof(u.sampler) <: Upsample2D
        load_state!(u.sampler.conv, state.upsamplers[1].conv)
    end
end

function load_state!(d::DownBlock2D, state)
    load_state!(d.resnets, state.resnets)
    if typeof(d.sampler) <: Downsample2D
        load_state!(d.sampler.conv, state.downsamplers[1].conv)
    end
end

function load_state!(t::TimestepEmbedding, state)
    load_state!(t.linear1, state.linear_1)
    load_state!(t.linear2, state.linear_2)
end

load_state!(::Flux.Dropout, _) = return

load_state!(::Nothing, _) = return

function load_state!(transformer::CLIPTextTransformer, state)
    load_state!(transformer.embeddings, state.embeddings)
    load_state!(transformer.encoder, state.encoder)
    load_state!(transformer.final_layer_norm, state.final_layer_norm)
end

function load_state!(encoder::CLIPEncoder, state)
    load_state!(encoder.layers, state.layers)
end

function load_state!(layer::CLIPEncoderLayer, state)
    for key in keys(state)
        load_state!(getfield(layer, key), getfield(state, key))
    end
end

function load_state!(attn::CLIPAttention, state)
    load_state!(attn.q, state.q_proj)
    load_state!(attn.k, state.k_proj)
    load_state!(attn.v, state.v_proj)
    load_state!(attn.out, state.out_proj)
end

function load_state!(mlp::CLIPMLP, state)
    load_state!(mlp.fc1, state.fc1)
    load_state!(mlp.fc2, state.fc2)
end

function load_state!(emb::CLIPTextEmbeddings, state)
    load_state!(emb.token_embedding, state.token_embedding)
    load_state!(emb.position_embedding, state.position_embedding)
end

function load_state!(emb::Embedding, state)
    copy!(emb.weights, transpose(state.weight))
end
