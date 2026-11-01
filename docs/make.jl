using BlackBoxComrade
using Documenter

DocMeta.setdocmeta!(BlackBoxComrade, :DocTestSetup, :(using BlackBoxComrade); recursive = true)

makedocs(;
    modules = [BlackBoxComrade],
    authors = "Paul Tiede <ptiede91@gmail.com> and contributors",
    sitename = "BlackBoxComrade.jl",
    format = Documenter.HTML(;
        canonical = "https://bh-cat.github.io/BlackBoxComrade.jl",
        edit_link = "main",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo = "github.com/BH-CAT/BlackBoxComrade.jl",
    devbranch = "main",
)
