# Diffusers.jl

An implementation of the subset of Hugging Face Diffusers in pure Julia.
Provides text-to-image StableDiffusion implementation.

|Painting of a farmer in the field|
|-|
|![painting-of-a-farmer-in-the-field-2](https://user-images.githubusercontent.com/17990405/233843029-2b8e1c22-51c1-4782-bdfd-b16177065bea.png)|

|Painting of a Dome Da Vinchi|Painting of a Dome Da Vinchi|Painting of a Dome Van Gogh|Painting of a Dome Van Gogh|
|-|-|-|-|
|![painting-of-a-dome-leonardo-da-vinchi-2](https://user-images.githubusercontent.com/17990405/234356486-7c15f543-f805-4f04-9eba-5e72a561b195.png)|![painting-of-a-dome-leonardo-da-vinchi-1](https://user-images.githubusercontent.com/17990405/234356507-5f78a5ca-fd9c-41ae-8fe8-53545ed652d1.png)|![painting-of-a-dome-van-gogh-2](https://user-images.githubusercontent.com/17990405/234356529-6505d3b1-63e3-4480-9ff9-e4393fe51e17.png)|![painting-of-a-dome-van-gogh-2 (copy)](https://user-images.githubusercontent.com/17990405/234356543-dfb95537-d41b-4606-bf16-02233b321974.png)|

|Refractive spheres sunlight|Refractive spheres|Thunder in mountains dark clouds|Edo era computer|
|-|-|-|-|
|![refractive-spheres-sunlight-1](https://user-images.githubusercontent.com/17990405/234356847-b057a1ec-ff2f-41a3-9f90-3195f0dbbb06.png)|![refractive-spheres-1](https://user-images.githubusercontent.com/17990405/234356875-304049c2-60aa-42a6-a717-c71d1e75031f.png)|![thunder-in-mountains-dark-clouds-1](https://github.com/JuliaNeuralGraphics/Diffusers.jl/assets/17990405/ff0c0993-8f20-44f8-b892-0bdbf8da4d1a)|![edo-era-computer-1](https://github.com/JuliaNeuralGraphics/Diffusers.jl/assets/17990405/90e9c46d-239a-4d32-8d7f-9d424c0aab58)|

### Installation

1. Clone the repo.
2. Launch Julia REPL from Diffusers.jl directory:
```bash
julia --threads=auto --project=.
```
3. Instantiate & update with `]up` command.

### Usage

```julia
julia> using AMDGPU # For AMD GPU support
julia> using CUDA   # For Nvidia GPU support

julia> using Flux
julia> Flux.gpu_backend!("AMDGPU") # For AMD GPU support
julia> Flux.gpu_backend!("CUDA") # For Nvidia GPU support

julia> using Diffusers
julia> Diffusers.main(["painting of a farmer in the field"]; device=gpu, precision=f16)
```

Images are saved in the Diffusers.jl directory.
