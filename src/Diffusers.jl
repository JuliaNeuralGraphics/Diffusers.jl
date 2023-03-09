module Diffusers

import MLUtils
import Transformers

using Flux

const Maybe{T} = Union{Nothing, T}

include("feed_forward.jl")
include("attention.jl")
include("transformer.jl")
include("resnet.jl")

include("schedulers/pndm.jl")

include("load_utils.jl")

# TODO
# - UNet2DCondition: https://github.com/huggingface/diffusers/blob/c7da8fd23359a22d0df2741688b5b4f33c26df21/src/diffusers/models/unet_2d_condition.py#L249
# - get_down_block: https://github.com/huggingface/diffusers/blob/c7da8fd23359a22d0df2741688b5b4f33c26df21/src/diffusers/models/unet_2d_blocks.py#LL89C29-L89C29
# - cross attention down block: https://github.com/huggingface/diffusers/blob/c7da8fd23359a22d0df2741688b5b4f33c26df21/src/diffusers/models/unet_2d_blocks.py#L770

function main()
    scheduler = PNDMScheduler(4; n_train_steps=1000)
    @show ndims(scheduler)

    # Where step is called in SD:
    # https://github.com/huggingface/diffusers/blob/c7da8fd23359a22d0df2741688b5b4f33c26df21/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#LL680C53-L680C53

    x = rand(Float32, 28, 28, 3, 1)
    sample = rand(Float32, 28, 28, 3, 1)

    set_timesteps!(scheduler, 100)
    for i in 1:100
        step(scheduler, x; t=(i - 1), sample)
    end

    return
end

end
