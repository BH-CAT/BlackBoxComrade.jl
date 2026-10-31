# BlackBoxComrade

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://blackholeseh.github.io/BlackBoxComrade.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://blackholeseh.github.io/BlackBoxComrade.jl/dev/)
[![Build Status](https://github.com/BlackHolesEh/BlackBoxComrade.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/BlackHolesEh/BlackBoxComrade.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/BlackHolesEh/BlackBoxComrade.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/BlackHolesEh/BlackBoxComrade.jl)

BlackBoxComrade is a Bayesian VLBI imaging pipeline built on
[Comrade.jl](https://github.com/ptiede/Comrade.jl). Comrade provides the instrument model,
the posterior and the sampler interfaces, with sky models from its companion packages
[VLBISkyModels.jl](https://github.com/EHTJulia/VLBISkyModels.jl) and
[VLBIImagePriors.jl](https://github.com/ptiede/VLBIImagePriors.jl). This package assembles a
Comrade posterior from TOML files, runs the optimization and sampling, and writes the images,
calibration tables and diagnostics.

A run is described by **four TOML config files**:

1. **image** — sky model: grid, mean model, polarization representation, random-field order,
   total flux.
2. **instrument** — gain/leakage parameterization and per-parameter priors, assembled
   generically from the TOML (no hand-written builder per case).
3. **data** — which observation to load (UVFITS or HOPS `.dlist`), averaging/noise, and the
   flag table.
4. **fitting** — optimizer, the noise-tempering schedule, and the MCMC sampler
   (AdvancedHMC NUTS, Reactant NUTS, or multi-chain MCLMC from
   [KineticSamplers.jl](https://github.com/BlackHolesEh/KineticSamplers.jl)).

See [`examples/`](examples) for annotated configs.

## Running

The package provides the `bbcomrade` [app](https://pkgdocs.julialang.org/v1/apps/) (Julia ≥ 1.12).
Install it once, with `~/.julia/bin` on your `PATH`:

```julia-repl
pkg> app add https://github.com/BlackHolesEh/BlackBoxComrade.jl
```

(`pkg> app develop path/to/BlackBoxComrade` for a local checkout). Then

```bash
bbcomrade --threads=4 -- \
    --image examples/image.toml \
    --instrument examples/instrument_mixed.toml \
    --data examples/data.toml \
    --fitting examples/fitting.toml \
    --outpath runs/m87
```

Julia flags such as `--threads` go before `--`, the app's options after it; `bbcomrade --help`
lists the options. The same command line runs without installing the app as
`julia --project=drivers -t 4 drivers/main.jl --image ...`.

Sampling on Reactant NUTS is selected inside the fitting TOML (`use_reactant = true`,
`sampler = "reactant"`), since that choice also governs the optimizer path. To resume a
previous run from its serialized optimum, add the one-off `--restart`:

```bash
bbcomrade --threads=4 -- --restart \
    --image examples/image.toml \
    --instrument examples/instrument_mixed.toml \
    --data examples/data.toml \
    --fitting examples/fitting.toml \
    --outpath runs/m87
```

Each run writes the MAP image (`_optimal.fits`/`.png`), residual plots, gain/leakage
caltables (`_ctable_*.csv`/`.png`), a serialized optimum (`_optimum_allres.jls`), and a set
of posterior image draws under `images/`.

## Using as a library

```julia
using BlackBoxComrade, TOML
skym, imgdata = build_sky_config(TOML.parsefile("examples/image.toml"))
intm          = build_instrument_config(TOML.parsefile("examples/instrument_mixed.toml"))
dcoh          = build_data_config(TOML.parsefile("examples/data.toml"))
strategy      = build_fitting_config(TOML.parsefile("examples/fitting.toml"))
comrade_imager("runs/m87", skym, intm, dcoh; strategy, imgdata)
```

See [`CLAUDE.md`](CLAUDE.md) for the source layout and architecture.
