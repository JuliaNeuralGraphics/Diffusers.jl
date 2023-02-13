
# torch.ModuleList
# Actual implementation
# https://github.com/chengchingwen/Transformers.jl/blob/8de6eb7d3f03b94fb1077588a960803f37f08496/src/huggingface/models/base.jl#L110
# MIT License

struct ModuleList{N, T<:Tuple}
    _modules::T
    ModuleList(ms) = new{length(ms), typeof(ms)}(ms)
end
  
ModuleList(ms...) = ModuleList(ms)
ModuleList(ms::Vector) = ModuleList(Tuple(ms))

Functors.functor(::Type{<:ModuleList}, modulelist) = modulelist._modules, y -> ModuleList(y)

Base.iterate(modulelist::ModuleList) = iterate(modulelist._modules)
Base.iterate(modulelist::ModuleList, i...) = iterate(modulelist._modules, i...)
Base.length(::ModuleList{N}) where N = N
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
        if (typeof(layerᵢ) <: Flux.Dropout)
            println("Flux.Dropout is not loaded.")
        else 
            load_state!(layerᵢ, state[i]) 
        end
    end
end  