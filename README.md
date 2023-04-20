# Diffusers.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaNeuralGraphics.github.io/Diffusers.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaNeuralGraphics.github.io/Diffusers.jl/dev)
[![Build Status](https://github.com/JuliaNeuralGraphics/Diffusers.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaNeuralGraphics/Diffusers.jl/actions/workflows/CI.yml?query=branch%3Amain)

![painting-of-a-farmer-in-the-field](https://user-images.githubusercontent.com/17990405/233398391-40dfc481-c05d-44e4-9535-b31b86b2dcf1.png)

*"Painting of a farmer in the field"*

### Usage

```julia
using Revise
using Diffusers
Diffusers.main()
```

Edit prompts in the `Diffusers.main` function and re-run it to generate new images.

### GPU selection

- AMDGPU:

Create `LocalPreferences.toml` file in Diffusers.jl directory with the following content:

```toml
[AMDGPU]
use_artifacts = false

[Flux]
gpu_backend = "AMD"
```

Double check that AMDGPU uses system ROCm instead of artifacts by calling `AMDGPU.versioninfo()`.

- CUDA: not yet supported.
 
