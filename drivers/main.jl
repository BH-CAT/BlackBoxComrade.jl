# The `bbcomrade` command line run from this environment: `julia --project=drivers drivers/main.jl
# --image ... --instrument ... --data ... --fitting ...` (`--help` for the options).
using Pkg
Pkg.activate(@__DIR__)

using BlackboxComrade

BlackboxComrade.bbcomrade(ARGS)
