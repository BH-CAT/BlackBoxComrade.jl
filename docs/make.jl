using BlackBoxComrade
using Documenter

DocMeta.setdocmeta!(BlackBoxComrade, :DocTestSetup, :(using BlackBoxComrade); recursive = true)

makedocs(;
    modules = [BlackBoxComrade],
    authors = "Paul Tiede <ptiede91@gmail.com> and contributors",
    sitename = "BlackBoxComrade.jl",
    format = Documenter.HTML(;
        canonical = "https://blackholeseh.github.io/BlackBoxComrade.jl",
        edit_link = "main",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo = "github.com/BlackHolesEh/BlackBoxComrade.jl",
    devbranch = "main",
)
