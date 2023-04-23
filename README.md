# Diffusers.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaNeuralGraphics.github.io/Diffusers.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaNeuralGraphics.github.io/Diffusers.jl/dev)
[![Build Status](https://github.com/JuliaNeuralGraphics/Diffusers.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaNeuralGraphics/Diffusers.jl/actions/workflows/CI.yml?query=branch%3Amain)

![painting-of-a-farmer-in-the-field-2](https://user-images.githubusercontent.com/17990405/233843029-2b8e1c22-51c1-4782-bdfd-b16177065bea.png)

*"Painting of a farmer in the field"*

### Installation

1. Clone the repo.
2. Launch Julia REPL from Diffusers.jl directory: `julia --threads=auto --project=.`
3. Instantiate & update with `]up` command.
4. **NOTE:** check [AMDGPU.jl Navi 2 support](https://amdgpu.juliagpu.org/dev/#Navi-2-(GFX103x)-support) section.

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
 
