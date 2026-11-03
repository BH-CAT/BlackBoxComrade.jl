# BlackboxComrade

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://bh-cat.github.io/BlackboxComrade.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://bh-cat.github.io/BlackboxComrade.jl/dev/)
[![Build Status](https://github.com/BH-CAT/BlackboxComrade.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/BH-CAT/BlackboxComrade.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/BH-CAT/BlackboxComrade.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/BH-CAT/BlackboxComrade.jl)

BlackboxComrade is a Bayesian VLBI imaging pipeline built on
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
   (AdvancedHMC NUTS, Reactant NUTS, or multi-chain MCLMC or WALNUTS from
   [KineticSamplers.jl](https://github.com/BH-CAT/KineticSamplers.jl)).

See [`examples/`](examples) for annotated configs.

## Installation

BlackboxComrade needs Julia 1.12 or later, which
[juliaup](https://github.com/JuliaLang/juliaup) installs.

BlackboxComrade and its dependency
[KineticSamplers.jl](https://github.com/BH-CAT/KineticSamplers.jl) are registered in
[BHCATRegistry](https://github.com/BH-CAT/BHCATRegistry), the BH-CAT registry, and both
KineticSamplers and the registry are private. Members of the [BH-CAT](https://github.com/BH-CAT)
GitHub organization can read all of its repositories; to get access, ask an organization
owner to invite you. Pkg also needs Git credentials it can use. One way is the GitHub CLI as
Git's credential helper, with Pkg running the system `git`:

```bash
gh auth login
gh auth setup-git
export JULIA_PKG_USE_CLI_GIT=true
```

The package provides the `bbcomrade` [app](https://pkgdocs.julialang.org/v1/apps/). From
the Julia package REPL (press `]` at the `julia>` prompt), add the registry once and install
the app:

```julia-repl
pkg> registry add https://github.com/BH-CAT/BHCATRegistry.git
pkg> app add BlackboxComrade
```

This puts `bbcomrade` in `~/.julia/bin`, which must be on your `PATH`; `bbcomrade --help`
checks the installation. `pkg> app update bbcomrade` updates it and `pkg> app rm bbcomrade`
removes it. For a local checkout, use `pkg> app develop path/to/BlackboxComrade` instead.

## Running

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

The sampler is selected inside the fitting TOML (`[sampler] algorithm`, and
`[run] use_reactant = true` for Reactant NUTS, MCLMC or WALNUTS), since Reactant also governs
the optimizer path. On the Reactant path, `[run] device = "cpu"` or `"gpu"` selects the
backend; left out, Reactant uses a GPU when one is available. To resume a
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
using BlackboxComrade, TOML
skym, imgdata = build_sky_config(TOML.parsefile("examples/image.toml"))
intm          = build_instrument_config(TOML.parsefile("examples/instrument_mixed.toml"))
dcoh          = build_data_config(TOML.parsefile("examples/data.toml"))
strategy      = build_fitting_config(TOML.parsefile("examples/fitting.toml"))
comrade_imager("runs/m87", skym, intm, dcoh; strategy, imgdata)
```

See [`CLAUDE.md`](CLAUDE.md) for the source layout and architecture.

## Releasing

Bump `version` in `Project.toml` and merge it. Then comment `/register` on any issue or pull
request to release the head of `main`. The Register workflow checks that the commenter has
write access, registers the version in BHCATRegistry, creates the `vX.Y.Z` release and
replies with the result.
