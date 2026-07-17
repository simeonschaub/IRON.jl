using IRON
using Documenter

DocMeta.setdocmeta!(IRON, :DocTestSetup, :(using IRON); recursive = true)

makedocs(;
    modules = [IRON],
    authors = "Simeon David Schaub <simeon@schaub.rocks> and contributors",
    sitename = "IRON.jl",
    format = Documenter.HTML(;
        canonical = "https://simeonschaub.github.io/IRON.jl",
        edit_link = "main",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo = "github.com/simeonschaub/IRON.jl",
    devbranch = "main",
)
