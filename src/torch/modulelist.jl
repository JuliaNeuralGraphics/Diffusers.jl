
# torch.ModuleList
# Actual implementation
# https://github.com/chengchingwen/Transformers.jl/blob/8de6eb7d3f03b94fb1077588a960803f37f08496/src/huggingface/models/base.jl#L110
# MIT License

"""
    torch.ModuleList(modules)
ModuleList is an indexed collection of Flux or torch modules. Users must write the forward definition of the ModuleList. 

# Examples
```
julia> state_dict, cfg = load_pretrained_model("runwayml/stable-diffusion-v1-5", "unet/config.json", "unet/diffusion_pytorch_model.bin") 
julia> ml = torch.ModuleList([torch.Linear(rand(320, 320), rand(320, 1)), Flux.Dropout(0.1)])
julia> torch.load_state!(ml, state_dict.down_blocks[1].attentions[1].transformer_blocks[1].attn1.to_out)
```
"""
struct ModuleList
    _modules::Tuple
end
  
ModuleList(ms...) = ModuleList(ms)
ModuleList(ms::Vector) = ModuleList(Tuple(ms))

Functors.functor(::Type{<:ModuleList}, modulelist) = modulelist._modules, y -> ModuleList(y)

Base.iterate(modulelist::ModuleList) = iterate(modulelist._modules)
Base.iterate(modulelist::ModuleList, i...) = iterate(modulelist._modules, i...)
Base.length(modulelist::ModuleList) = length(modulelist._modules)
Base.getindex(modulelist::ModuleList, i) = modulelist._modules[i]
  
function get_state_dict(state, prefix, modulelist::ModuleList)
    param = Functors.functor(modulelist)[1]
    for (i, layer) in enumerate(modulelist)
        cprefix = join((prefix, i-1), '.')
        get_state_dict(state, cprefix, layer)
    end
end
  
function load_state!(layer::ModuleList, state)
    for (i, layerᵢ) in enumerate(layer)
        # TODO: handle cases where layer is missing
        if (typeof(layerᵢ) <: Dropout)
            println("WARN: Flux.Dropout is not loaded as there are no parameters in the state_dict.")
        else
            load_state!(layerᵢ, state[i]) 
        end
    end
end