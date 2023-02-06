using Diffusers
using Documenter

DocMeta.setdocmeta!(Diffusers, :DocTestSetup, :(using Diffusers); recursive=true)

makedocs(;
    modules=[Diffusers],
    authors="Harish Anand",
    repo="https://github.com/JuliaNeuralGraphics/Diffusers.jl/blob/{commit}{path}#{line}",
    sitename="Diffusers.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://JuliaNeuralGraphics.github.io/Diffusers.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/JuliaNeuralGraphics/Diffusers.jl",
    devbranch="main",
)
