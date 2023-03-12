Base.@kwdef mutable struct PNDMScheduler{
    A <: AbstractVector{Float32},
    T <: AbstractVector{Int},
    S <: AbstractArray{Float32},
}
    const α̂::A # Same as α̅ but lives on the current device.
    const α::Vector{Float32}
    const α̅::Vector{Float32}
    const β::Vector{Float32}

    const α_final::Float32
    const σ₀::Float32 = 1f0 # Standard deviation of the initial noise distribution.
    const skip_prk_steps::Bool
    const step_offset::Int

    # Adjustable.

    timesteps::T = Int[]
    prk_timesteps::Vector{Int} = Int[]
    plms_timesteps::Vector{Int} = Int[]
    _timesteps::Vector{Int}

    # Counters.

    x_current::S
    sample::S
    xs::Vector{S}

    n_train_timesteps::Int
    n_inference_timesteps::Int = -1
    counter::Int = 0
end

function PNDMScheduler(nd::Int;
    β_schedule::Symbol = :linear,
    β_range::Pair{Float32, Float32} = 1f-4 => 2f-2,
    n_train_timesteps::Int = 1000,
    α_to_one::Bool = false,
    skip_prk_steps::Bool = false,
    step_offset::Int = 0,
)
    β = if β_schedule == :linear
        β_step = (β_range[2] - β_range[1]) / (n_train_timesteps - 1)
        collect(β_range[1]:β_step:β_range[2])
    elseif β_schedule == :scaled_linear
        β_start, β_end = √β_range[1], √β_range[2]
        β_step = (β_end - β_start) / (n_train_timesteps - 1)
        collect(β_start:β_step:β_end) .^ 2
    else
        throw(ArgumentError("""
        Unsupported β schedule mode: `$β_schedule`.
        Supported modes: `:linear`, `:scaled_linear`.
        """))
    end

    α = 1f0 .- β
    α̅ = cumprod(α)
    α_final = α_to_one ? 1f0 : α̅[1]

    _timesteps = (n_train_timesteps - 1):-1:0 |> collect

    x_current = zeros(Float32, ntuple(_ -> 1, Val(nd)))
    sample = Array{Float32, nd}(undef, ntuple(_ -> 0, Val(nd)))

    PNDMScheduler(;
        α̂=α̅, α, α̅, β, α_final, skip_prk_steps, step_offset,
        _timesteps,
        x_current, sample, xs=Array{Float32, nd}[],
        n_train_timesteps)
end

for T in FluxDeviceAdaptors
    @eval function Adapt.adapt_storage(to::$(T), pndm::PNDMScheduler)
        PNDMScheduler(
            Adapt.adapt(to, pndm.α̂),
            pndm.α, pndm.α̅, pndm.β,
            pndm.α_final, pndm.σ₀, pndm.skip_prk_steps, pndm.step_offset,
            Adapt.adapt(to, pndm.timesteps),
            pndm.prk_timesteps, pndm.plms_timesteps, pndm._timesteps,

            Adapt.adapt(to, pndm.x_current),
            Adapt.adapt(to, pndm.sample),
            [Adapt.adapt(to, x) for x in pndm.xs],

            pndm.n_train_timesteps, pndm.n_inference_timesteps, pndm.counter)
    end
end

Base.ndims(::PNDMScheduler{A, T, S}) where {A, T, S} = ndims(S)

"""
    set_timesteps!(pndm::PNDMScheduler, n_inference_timesteps::Int)

Set discrete timesteps used for the diffusion chain.

# Arguments:
- `n_inference_timesteps::Int`: Number of diffusion steps used when
    generating samples with pre-trained model.
"""
function set_timesteps!(pndm::PNDMScheduler, n_inference_timesteps::Int)
    # TODO @assert n_inference_timesteps ≤ n_training_steps
    pndm.n_inference_timesteps = n_inference_timesteps
    ratio = pndm.n_train_timesteps ÷ pndm.n_inference_timesteps
    pndm._timesteps =
        round.(Int, collect(0:(pndm.n_inference_timesteps - 1)) .* ratio) .+
        pndm.step_offset

    if pndm.skip_prk_steps
        empty!(pndm.prk_timesteps)
        pndm.plms_timesteps = reverse(cat(
            pndm._timesteps[1:end - 1],
            pndm._timesteps[2:end - 1],
            pndm._timesteps[2:end]; dims=1))
    else
        pndm_order = 4
        prk_timesteps =
            repeat(pndm._timesteps[end - pndm_order + 1:end]; inner=2) .+
            repeat([0, ratio ÷ 2], pndm_order)
        prk_timesteps = reverse(
            repeat(prk_timesteps[1:end - 1]; inner=2)[2:end - 1])

        pndm.prk_timesteps = prk_timesteps
        pndm.plms_timesteps = reverse(pndm._timesteps[1:end - 3])
    end

    copy!(pndm.timesteps, cat(pndm.prk_timesteps, pndm.plms_timesteps; dims=1))

    # Reset counters.
    pndm.x_current = similar(pndm.x_current, ntuple(_ -> 1, Val(ndims(pndm))))
    fill!(pndm.x_current, 0f0)
    empty!(pndm.xs)
    pndm.counter = 0
    return
end

"""
Predict the sample at the previous `t - 1` timestep by reversing the SDE.

# Arguments:

- `x`: Output from diffusion model (most often the predicted noise).
- `t::Int`: Current discrete timestep in the diffusion chain.
- `sample`: Sample at the current timestep `t` created by diffusion process.
"""
function step!(pndm::PNDMScheduler{A, T, S}, x::S; t::Int, sample::S) where {
    A, T, S,
}
    (pndm.counter < length(pndm.prk_timesteps) && !pndm.skip_prk_steps) ?
        step_prk!(pndm, x; t, sample) :
        step_plms!(pndm, x; t, sample)
end

"""
Step function propagating the sample with the Runge-Kutta method.
RK takes 4 forward passes to approximate the solution
to the differential equation.
"""
function step_prk!(pndm::PNDMScheduler{A, T, S}, x::S; t::Int, sample::S) where {
    A, T, S,
}
    ratio = pndm.n_train_timesteps ÷ pndm.n_inference_timesteps
    δt = (pndm.counter % 2 == 0) ? (ratio ÷ 2) : 0
    prev_t, t = t - δt, pndm.prk_timesteps[(pndm.counter ÷ 4) * 4 + 1]

    if pndm.counter % 4 == 0
        # Re-assign to get correct shape.
        pndm.x_current = pndm.x_current .+ (1f0 / 6f0) .* x
        pndm.sample = sample
        push!(pndm.xs, x)
    elseif (pndm.counter - 1) % 4 == 0
        pndm.x_current .+= (1f0 / 3f0) .* x
    elseif (pndm.counter - 2) % 4 == 0
        pndm.x_current .+= (1f0 / 3f0) .* x
    elseif (pndm.counter - 3) % 4 == 0
        x = pndm.x_current .+ (1f0 / 6f0) .* x
        fill!(pndm.x_current, 0f0)
    end

    pndm.counter += 1
    previous_sample(pndm, x; t, prev_t, sample=pndm.sample)
end

"""
Step function propagating the sample with the linear multi-step method.
Has one forward pass with multiple times to approximate the solution.
"""
function step_plms!(pndm::PNDMScheduler{A, T, S}, x::S; t::Int, sample::S) where {
    A, T, S,
}
    !pndm.skip_prk_steps && length(pndm.xs) < 3 && error("""
    Linear multi-step method can only be run after at least 12 steps in PRK
    mode and has sampled `3` forward passes.
    Current amount is `$(length(pndm.xs))`.
    """)

    ratio = pndm.n_train_timesteps ÷ pndm.n_inference_timesteps
    prev_t = t - ratio

    if pndm.counter == 1
        prev_t, t = t, t + ratio
    else
        if !isempty(pndm.xs)
            min_idx = max(1, length(pndm.xs) - 2)
            pndm.xs = pndm.xs[min_idx:end]
        end
        push!(pndm.xs, x)
    end

    if length(pndm.xs) == 1 && pndm.counter == 0
        pndm.sample = sample
    elseif length(pndm.xs) == 1 && pndm.counter == 1
        x = (x .+ pndm.xs[end]) .* 0.5f0
        sample = pndm.sample
    elseif length(pndm.xs) == 2
        x = (3f0 .* pndm.xs[end] .- pndm.xs[end - 1]) .* 0.5f0
    elseif length(pndm.xs) == 3
        x = (1f0 / 12f0) .* (
            23f0 .* pndm.xs[end] .- 16f0 .* pndm.xs[end - 1] .+
            5f0 .* pndm.xs[end - 2])
    else
        x = (1f0 / 24f0) .* (
            55f0 .* pndm.xs[end] .- 59f0 .* pndm.xs[end - 1] .+
            37f0 .* pndm.xs[end - 2] .- 9f0 .* pndm.xs[end - 3])
    end

    pndm.counter += 1
    previous_sample(pndm, x; t, prev_t, sample)
end

"""
# Arguments:

- `x`: Sample for which to apply noise.
- `ξ`: Noise which to apply to `x`.
- `timesteps`: Vector of discrete timesteps starting at `0`.
"""
function add_noise(pndm::PNDMScheduler{A, T, S}, x::S, ξ::S, timesteps::T) where {
    A, T, S,
}
    αᵗ = reshape(pndm.α̂[timesteps .+ 1], ntuple(_->1, Val(ndims(S) - 1))..., :)
    α̅, β̅ = sqrt.(αᵗ), sqrt.(1f0 .- αᵗ)
    α̅ .* x .+ β̅ .* ξ
end

# Equation (9) from paper.
function previous_sample(
    pndm::PNDMScheduler{A, T, S}, x::S; t::Int, prev_t::Int, sample::S,
) where {A, T, S}
    αₜ₋ᵢ, αₜ = (prev_t ≥ 0 ? pndm.α̅[prev_t + 1] : pndm.α_final), pndm.α̅[t + 1]
    βₜ₋ᵢ, βₜ = (1f0 - αₜ₋ᵢ), (1f0 - αₜ)

    γ = √(αₜ₋ᵢ / αₜ)
    ϵ = αₜ * √βₜ₋ᵢ + √(αₜ₋ᵢ * αₜ * βₜ)
    γ .* sample .- (αₜ₋ᵢ - αₜ) .* x ./ ϵ
end

# HGF integration.

"""
Create PNDMScheduler from HuggingFace config file.

# Arguments:

- `nd::Int`: Number of the dimensions of the samples (e.g. 4 for images).
- `model_name::String`: Name of the model.
- `filename::String`: Path to a config file in the model's repo.

# Example:

```jldoctest
julia> pndm = Diffusers.PNDMScheduler(HGF, 4;
    model_name="runwayml/stable-diffusion-v1-5",
    filename="scheduler/scheduler_config.json");
```
"""
function PNDMScheduler(::Val{:HGF}, nd::Int; model_name::String, filename::String)
    config = Diffusers.load_hgf_config(model_name; filename)
    PNDMScheduler(nd;
        β_schedule=Symbol(config["beta_schedule"]),
        β_range=Float32(config["beta_start"]) => Float32(config["beta_end"]),
        n_train_timesteps=config["num_train_timesteps"],
        skip_prk_steps=config["skip_prk_steps"],
        α_to_one=config["set_alpha_to_one"])
end
