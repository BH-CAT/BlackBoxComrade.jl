using BlackboxComrade
using Documenter

DocMeta.setdocmeta!(BlackboxComrade, :DocTestSetup, :(using BlackboxComrade); recursive = true)

makedocs(;
    modules = [BlackboxComrade],
    authors = "Paul Tiede <ptiede91@gmail.com> and contributors",
    sitename = "BlackboxComrade.jl",
    format = Documenter.HTML(;
        canonical = "https://bh-cat.github.io/BlackboxComrade.jl",
        edit_link = "main",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo = "github.com/BH-CAT/BlackboxComrade.jl",
    devbranch = "main",
)
