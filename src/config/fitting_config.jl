# The fitting-strategy config: optimization, the noise-tempering schedule, and the MCMC
# sampler (AdvancedHMC NUTS, Reactant NUTS, or KineticSamplers MCLMC or WALNUTS).

const KINETIC_ALGORITHMS = ("mclmc", "walnuts")
const KINETIC_WARMUP = Dict("mclmc" => 150_000, "walnuts" => 1_000)
const MOVE_KINDS = ("flux_gain", "field_scale", "rho_field", "mean_field", "phase_sheet", "chain_hyper")

"""
    MoveSpec

One `[[sampler.moves]]` table of the fitting config: the move `kind` (one of
`$(MOVE_KINDS)`), the parameters it acts on (`params`, `nothing` for all that apply), the
proposals per call (`rounds`), the warmup acceptance target, and the initial step scale of a
random-walk move (`nothing` for the kind's default). See [`build_moves`](@ref).
"""
Base.@kwdef struct MoveSpec
    kind::String
    params::Union{Nothing, Vector{String}} = nothing
    rounds::Int = 1
    target_accept::Float64 = 0.45
    initial_scale::Union{Nothing, Float64} = nothing
    band_limit::Float64 = Inf
end

"""
    FittingStrategy

Typed description of how to fit: the optimizer, the fractional-noise tempering schedule,
the MCMC sampler and its tuning, and run-level flags (Reactant, benchmark, start).

Note: `restart` is deliberately NOT part of the strategy — resuming a run is a one-off
action passed at call time (the `--restart` CLI flag / `comrade_imager(...; restart=true)`),
not configuration to track in a TOML.
"""
Base.@kwdef struct FittingStrategy
    # optimizer
    opt_method::String = "Adam"     # Adam | AdamW | LBFGS (LBFGS is CPU-only)
    maxiters::Int = 10_000
    ntrials::Int = 5
    g_tol::Float64 = 0.1
    eta::Float64 = 0.001            # learning rate for the Optimisers.jl rules (Adam/AdamW)
    # sky scale parameters held at their prior medians during optimization; non-empty also
    # rescales the optimum's fields to unit rms (see `build_fitting_config`)
    fix_scales::Vector{Symbol} = Symbol[]
    # tempering: fractional-noise level per optimization round (0.0 = full data)
    noise_schedule::Vector{Float64} = [0.05, 0.025, 0.0]
    nsample::Int = 10_000
    nadapt::Int = 5_000
    step_size::Float64 = 0.01
    target_accept::Float64 = 0.9
    init_buffer::Int = 200
    term_buffer::Int = 500
    max_tree_depth::Int = 10
    chunk_size::Int = 100
    # "nuts", or "mclmc" or "walnuts" for KineticSamplers chains with `chains` chains and a
    # pooled warmup
    algorithm::String = "nuts"
    chains::Int = 4
    max_step_halvings::Int = 3
    # Metropolis–Hastings moves run between sampler chunks or blocks (see `build_moves`).
    moves::Vector{MoveSpec} = MoveSpec[]
    # run
    # Latent space optimized and sampled in: "flat" (`asflat`) or "stdnormal" (the
    # ProbabilityTransports `StdNormal` transport, where the prior is N(0, I); needed by
    # priors with no flat transform, e.g. `AngularProjectedNormal`).
    latent_space::String = "flat"
    use_reactant::Bool = false
    # Reactant backend, "cpu" or "gpu"; `nothing` keeps Reactant's default (a GPU when one
    # is available).
    device::Union{Nothing, String} = nothing
    benchmark::Bool = true
    # On the Reactant path, check the device log-density+gradient against the CPU/Enzyme
    # reference at a prior draw before optimizing, and error on mismatch (catches broken
    # device models that silently produce garbage fits).
    verify_reactant::Bool = false
    start::Union{Nothing, String} = nothing
    # Reactant checkpointing (FITS + PNG + residuals): render every `sample_checkpoint`
    # samples (this is also the sampling DiskStore stride). Warmup runs in chunks of the same
    # stride, rendering the current draw each chunk, and is checkpointed every
    # `warmup_checkpoint` steps (so warmup is resumable via `restart`). 0 disables it.
    # Optimization is deliberately NOT checkpointed — rendering on the host each step is far
    # slower than the device step.
    sample_checkpoint::Int = 0
    # Warmup steps between state checkpoints on the Reactant path (`checkpoint_every` of
    # `sample`); `nothing` keeps Comrade's default.
    warmup_checkpoint::Union{Nothing, Int} = nothing
end

"""
    build_fitting_config(cfg::AbstractDict) -> FittingStrategy

Parse a fitting-strategy TOML into a [`FittingStrategy`](@ref). Sections: `[optimizer]`,
`[tempering]`, `[sampler]`, `[run]`. `sampler.algorithm` is `"nuts"` (default), whose backend
(AdvancedHMC NUTS vs Reactant NUTS with its diagonal metric) follows `run.use_reactant`, or
`"mclmc"` or `"walnuts"`, which run `sampler.chains` chains of KineticSamplers MCLMC or WALNUTS
(see [`sample_kinetic`](@ref)) and need `run.use_reactant = true`. `nadapt` is the warmup length per chain, in steps for
MCLMC (default 150 000) and in draws for WALNUTS (default 1000), and `nsample` the draws kept
per chain (default 1000). For MCLMC a draw is kept every `chunk_size` steps, and `max_step_halvings` (default 3) is the number of halvings
of a step whose energy error is too large. For WALNUTS every draw is kept, `chunk_size`
(default 10) is the draws per sampling block, which must divide `nsample`, and `max_tree_depth`
(default 10) is the most trajectory doublings. `step_size` and the run checkpoints are
errors with either algorithm, as is `max_tree_depth` with MCLMC. `run.start` may also name the
run base of an MCLMC or WALNUTS run (the directory holding its `chain_k`), whose chains' last
draws start the chains one to one.

`run.device` (`"cpu"` or `"gpu"`) selects the Reactant backend and needs
`run.use_reactant = true`. Left out, Reactant uses a GPU when one is available and the CPU
otherwise. `"gpu"` is an error when no GPU is available.

`[optimizer] fix_scales` lists scalar sky parameters (e.g. `["σb", "σc", "σd"]`) that every
optimization stage holds at the median of their prior; the other parameters are optimized
as usual. After the last stage, every non-centered sky field `X` with scale `σX` (in
`polexp_srf`: `a`, `b`, `c`, `d`) is rescaled to `σX * rms(X)` and `X / rms(X)`, which
leaves the image unchanged and puts the white coefficients at the radius the prior's typical
set has (`rms(X) = 1`); see [`rescale_fields`](@ref). The sampler starts from that point
with every parameter free. Empty (the default) changes nothing. A name that is not a scalar
sky parameter of the model is an error when optimization starts.

`[[sampler.moves]]` tables list Metropolis–Hastings moves run between the Reactant NUTS
chunks (see [`build_moves`](@ref)), each table one move kind:

```toml
[[sampler.moves]]
kind = "mean_field"
params = ["fwhm"]     # optional: a subset of the parameters the kind acts on
rounds = 10           # optional: proposals of each move per call (default 1)
target_accept = 0.45  # optional: warmup acceptance target (default 0.45)
initial_scale = 0.01  # optional: initial random-walk step scale (default per kind)
band_limit = 3.0      # optional, field_scale and rho_field only (default inf), see below
```

The kinds, all of which leave the likelihood unchanged unless `band_limit` is finite:

  - `"flux_gain"`: total flux against a common shift of the gain log-amplitudes, `lg1μ` when
    the gain scheme has it and the `lg1` chain otherwise (no `params`);
  - `"field_scale"`: each non-centered sky field against its scale `σX` (`params`: fields);
  - `"rho_field"`: each spectral parameter of each stationary random field (Markov RF
    correlation lengths, Matérn outer scale and slope) against the field's white
    coefficients (`params`: fields);
    for these two kinds a finite `band_limit` compensates only the white coefficients with
    wavenumber up to `band_limit` times the longest baseline (stationary random fields
    only); the image then changes at higher wavenumbers and the move is accepted on the
    full posterior;
  - `"mean_field"`: each mean-model parameter against the log-intensity field `a` (PolExp
    stationary random-field models; `params`: mean-model parameters);
  - `"phase_sheet"`: a `±2π` shift of a real-line Gauss–Markov phase chain from a point on
    (`params`: chain terms; no `initial_scale`); `"stdnormal"` space only;
  - `"chain_hyper"`: each fitted hyperparameter field (`σ`, `τ`, `D`) of a Gauss–Markov
    instrument chain against the chain's whitened innovations, with the chain values fixed;
    one move per term and field, which steps every site's hyperparameter and accepts each
    site on its own, so a round is one sweep over the sites (`params`: chain terms);
    `"stdnormal"` space only.

A kind may appear more than once only with disjoint `params`. Moves run with the Reactant
NUTS, MCLMC and WALNUTS samplers (`use_reactant = true`), not with the CPU NUTS sampler.
Unknown kinds and keys are errors, as are the list form `moves = [...]` and
`moves_per_chunk`. With NUTS the moves run once per sampler chunk
(`chunk_size`, or `sample_checkpoint` when set); with MCLMC on every chain once per 50 blocks
of `chunk_size` steps, and with WALNUTS on every chain once per 10 blocks.

`[run] latent_space` picks the space the optimizer and sampler work in: `"flat"` (default,
`asflat`) or `"stdnormal"` (the `StdNormal` transport, where the prior is exactly N(0, I);
required by priors without a flat transform such as `AngularProjectedNormal`). In the
`"stdnormal"` space the start point's real-line phase chains are re-wrapped to their shortest
steps (see [`unwrap_phase_chains`](@ref)) before sampling.
"""
function build_fitting_config(cfg::AbstractDict)
    check_config_keys(
        cfg, ("optimizer", "tempering", "sampler", "run"),
        "the fitting config (top level)"
    )
    opt = get(cfg, "optimizer", Dict{String, Any}())
    temp = get(cfg, "tempering", Dict{String, Any}())
    samp = get(cfg, "sampler", Dict{String, Any}())
    run = get(cfg, "run", Dict{String, Any}())

    check_config_keys(
        opt, ("method", "maxiters", "ntrials", "g_tol", "eta", "fix_scales"), "[optimizer]"
    )
    check_config_keys(temp, ("noise_schedule",), "[tempering]")
    check_config_keys(
        samp,
        (
            "nsample", "nadapt", "step_size", "target_accept", "init_buffer",
            "term_buffer", "max_tree_depth", "chunk_size", "moves", "moves_per_chunk",
            "algorithm", "chains", "max_step_halvings",
        ),
        "[sampler]"
    )
    check_config_keys(
        run,
        (
            "use_reactant", "benchmark", "verify_reactant", "start", "sample_checkpoint",
            "checkpoint", "latent_space", "warmup_checkpoint", "device",
        ),
        "[run]"
    )

    use_reactant = Bool(get(run, "use_reactant", false))
    ahmc_only = filter(k -> haskey(samp, k), ["target_accept", "init_buffer", "term_buffer"])
    (use_reactant && !isempty(ahmc_only)) && error(
        "sampler.$(join(ahmc_only, ", sampler.")) apply to AdvancedHMC NUTS only " *
            "(run.use_reactant = false); Reactant NUTS has no setting for them"
    )
    device = haskey(run, "device") ? String(run["device"]) : nothing
    isnothing(device) || device in ("cpu", "gpu") ||
        error("unknown run.device '$device'. Allowed: cpu, gpu")
    (isnothing(device) || use_reactant) ||
        error("run.device needs run.use_reactant = true: without Reactant everything runs on the CPU")
    latent_space = String(get(run, "latent_space", "flat"))
    latent_space in ("flat", "stdnormal") ||
        error("unknown run.latent_space '$latent_space'. Allowed: flat, stdnormal")

    opt_method = String(get(opt, "method", "Adam"))
    opt_method in ("Adam", "AdamW", "LBFGS") ||
        error("unknown optimizer.method '$opt_method'. Allowed: Adam, AdamW, LBFGS")

    startval = get(run, "start", "")
    start = (startval == "") ? nothing : String(startval)

    fix_scales = get(opt, "fix_scales", String[])
    (fix_scales isa AbstractVector && all(v -> v isa AbstractString, fix_scales)) ||
        error("optimizer.fix_scales must be a list of sky parameter names, got $(repr(fix_scales))")
    allunique(fix_scales) || error("optimizer.fix_scales lists a name twice: $fix_scales")

    haskey(samp, "moves_per_chunk") && error(
        "sampler.moves_per_chunk is not a fitting-config key; set `rounds` in each [[sampler.moves]] table"
    )
    moves = _parse_move_specs(get(samp, "moves", Any[]), latent_space)
    isempty(moves) || use_reactant || error(
        "sampler.moves needs run.use_reactant = true: the CPU NUTS sampler does not run moves"
    )

    algorithm = String(get(samp, "algorithm", "nuts"))
    algorithm in ("nuts", KINETIC_ALGORITHMS...) || error(
        "sampler.algorithm must be \"nuts\", \"mclmc\" or \"walnuts\", got $(repr(algorithm))"
    )
    kinetic = algorithm in KINETIC_ALGORITHMS
    for (key, allowed) in (
            "step_size" => ("nuts",), "max_tree_depth" => ("nuts", "walnuts"),
            "chains" => KINETIC_ALGORITHMS, "max_step_halvings" => ("mclmc",),
        )
        haskey(samp, key) && !(algorithm in allowed) && error(
            "sampler.$key applies to sampler.algorithm = $(join(repr.(allowed), " or ")) only, " *
                "not $(repr(algorithm))"
        )
    end
    if kinetic
        use_reactant || error("sampler.algorithm = \"$algorithm\" needs run.use_reactant = true")
        run_nuts_only = filter(
            k -> get(run, k, 0) != 0, ["sample_checkpoint", "checkpoint", "warmup_checkpoint"]
        )
        isempty(run_nuts_only) ||
            error("run.$(join(run_nuts_only, ", run.")) apply to NUTS only")
    end
    chains = Int(get(samp, "chains", 4))
    chains >= 1 || error("sampler.chains must be at least 1, got $chains")
    nsample = Int(get(samp, "nsample", kinetic ? 1_000 : 10_000))
    chunk_size = Int(get(samp, "chunk_size", algorithm == "walnuts" ? 10 : 100))
    chunk_size >= 1 || error("sampler.chunk_size must be at least 1, got $chunk_size")
    (algorithm == "walnuts" && nsample % chunk_size != 0) && error(
        "sampler.nsample = $nsample must be a multiple of sampler.chunk_size = $chunk_size, " *
            "the WALNUTS draws per block"
    )
    return FittingStrategy(;
        opt_method = opt_method,
        maxiters = Int(get(opt, "maxiters", 10_000)),
        ntrials = Int(get(opt, "ntrials", 5)),
        g_tol = Float64(get(opt, "g_tol", 0.1)),
        eta = Float64(get(opt, "eta", 0.001)),
        fix_scales = Symbol.(fix_scales),
        noise_schedule = Float64.(get(temp, "noise_schedule", [0.05, 0.025, 0.0])),
        nsample,
        nadapt = Int(get(samp, "nadapt", get(KINETIC_WARMUP, algorithm, 5_000))),
        step_size = Float64(get(samp, "step_size", 0.01)),
        target_accept = Float64(get(samp, "target_accept", 0.9)),
        init_buffer = Int(get(samp, "init_buffer", 200)),
        term_buffer = Int(get(samp, "term_buffer", 500)),
        max_tree_depth = Int(get(samp, "max_tree_depth", 10)),
        chunk_size,
        moves,
        algorithm,
        chains,
        max_step_halvings = Int(get(samp, "max_step_halvings", 3)),
        latent_space = latent_space,
        use_reactant = use_reactant,
        device,
        benchmark = Bool(get(run, "benchmark", true)),
        verify_reactant = Bool(get(run, "verify_reactant", false)),
        start = start,
        # `checkpoint` is accepted as an alias for `sample_checkpoint`.
        sample_checkpoint = Int(get(run, "sample_checkpoint", get(run, "checkpoint", 0))),
        warmup_checkpoint = haskey(run, "warmup_checkpoint") ? Int(run["warmup_checkpoint"]) : nothing,
    )
end

# The `space` argument of `Comrade.maybe_transport` for a strategy: `nothing` is the flat space.
latent_space(strategy::FittingStrategy) = strategy.latent_space == "stdnormal" ? PT.StdNormal() : nothing

const _MOVE_KEYS = ("kind", "params", "rounds", "target_accept", "initial_scale", "band_limit")

function _parse_move_specs(tables, latent_space)
    (tables isa AbstractVector && all(t -> t isa AbstractDict, tables)) || error(
        "sampler.moves must be [[sampler.moves]] tables, each naming a move kind, e.g.\n" *
            "[[sampler.moves]]\nkind = \"mean_field\"\nrounds = 10\ngot $(repr(tables))"
    )
    specs = map(enumerate(tables)) do (i, t)
        where_ = "[[sampler.moves]] table $i"
        check_config_keys(t, _MOVE_KEYS, where_)
        kind = get(t, "kind", nothing)
        kind isa AbstractString || error("$where_ needs a kind (one of $(collect(MOVE_KINDS)))")
        kind in MOVE_KINDS || error("unknown move kind \"$kind\" in $where_. Allowed: $(collect(MOVE_KINDS))")
        params = get(t, "params", nothing)
        isnothing(params) || (params isa AbstractVector && !isempty(params) && all(p -> p isa AbstractString, params)) ||
            error("$where_: params must be a non-empty list of names, got $(repr(params))")
        rounds = get(t, "rounds", 1)
        (rounds isa Integer && rounds >= 1) || error("$where_: rounds must be an integer ≥ 1, got $(repr(rounds))")
        target_accept = get(t, "target_accept", 0.45)
        (target_accept isa Real && 0 < target_accept < 1) ||
            error("$where_: target_accept must lie in (0, 1), got $(repr(target_accept))")
        initial_scale = get(t, "initial_scale", nothing)
        isnothing(initial_scale) || (initial_scale isa Real && initial_scale > 0) ||
            error("$where_: initial_scale must be positive, got $(repr(initial_scale))")
        (kind == "phase_sheet" && !isnothing(initial_scale)) &&
            error("$where_: phase_sheet takes discrete ±2π steps and has no initial_scale")
        (kind == "flux_gain" && !isnothing(params)) && error("$where_: flux_gain takes no params")
        band_limit = get(t, "band_limit", Inf)
        (band_limit isa Real && band_limit > 0) || error(
            "$where_: band_limit must be a positive multiple of the longest baseline (inf for no limit), got $(repr(band_limit))"
        )
        (isfinite(band_limit) && !(kind in ("field_scale", "rho_field"))) &&
            error("$where_: band_limit applies only to field_scale and rho_field")
        (kind == "phase_sheet" && latent_space != "stdnormal") && error(
            "$where_: phase_sheet needs run.latent_space = \"stdnormal\" (the flat space's wrapped chains have no sheets)"
        )
        (kind == "chain_hyper" && latent_space != "stdnormal") && error(
            "$where_: chain_hyper needs run.latent_space = \"stdnormal\" (its sites are accepted without the likelihood)"
        )
        MoveSpec(;
            kind = String(kind), params = isnothing(params) ? nothing : String.(params),
            rounds = Int(rounds), target_accept = Float64(target_accept),
            initial_scale = isnothing(initial_scale) ? nothing : Float64(initial_scale),
            band_limit = Float64(band_limit),
        )
    end
    for kind in unique(s.kind for s in specs)
        same = [s.params for s in specs if s.kind == kind]
        length(same) == 1 && continue
        (any(isnothing, same) || !allunique(reduce(vcat, same))) && error(
            "sampler.moves lists kind \"$kind\" more than once with overlapping params; give each table disjoint params"
        )
    end
    return MoveSpec[specs...]
end
