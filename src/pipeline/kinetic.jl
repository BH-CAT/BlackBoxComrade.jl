# KineticSamplers MCLMC and WALNUTS chains in the run's latent space.

const KINETIC_START_JITTER = 0.001
const KINETIC_DRAWS_PER_FILE = 10
const KINETIC_RETURNED_DRAWS = 500
const KINETIC_LOG_DRAWS = 50
const MCLMC_MOVE_BLOCKS = 50
const WALNUTS_MOVE_BLOCKS = 10
const MCLMC_CHECK_BLOCKS = 10

"""
    LatentTarget(tpost, dim)

The LogDensityProblems interface of the device posterior `tpost` in its latent space, with
gradients and Hessian-vector products by Enzyme under Reactant.
"""
struct LatentTarget{P}
    tpost::P
    dim::Int
end

LogDensityProblems.capabilities(::Type{<:LatentTarget}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.dimension(t::LatentTarget) = t.dim
LogDensityProblems.logdensity(t::LatentTarget, x) = Comrade.logdensityof(t.tpost, x)

# `set_strong_zero` sends 0*Inf and 0*NaN to 0 in the derivative passes, without which a stiff
# image model has no finite gradient.
function LogDensityProblems.logdensity_and_gradient(t::LatentTarget, x)
    mode = Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal)
    derivs, val = Enzyme.gradient(mode, Comrade.logdensityof, Enzyme.Const(t.tpost), x)
    return val, last(derivs)
end

_latent_gradient(tpost, x) = last(LogDensityProblems.logdensity_and_gradient(LatentTarget(tpost, length(x)), x))

KineticSamplers.has_hessian_vector_product(::Type{<:LatentTarget}) = true

# Forward-mode differentiation of the reverse-mode gradient along `v`.
function KineticSamplers.logdensity_and_hessian_vector_product(t::LatentTarget, x, v)
    hv = only(
        Enzyme.autodiff(
            Enzyme.set_strong_zero(Enzyme.Forward), Enzyme.Const(_latent_gradient),
            Enzyme.Const(t.tpost), Enzyme.Duplicated(x, v)
        )
    )
    return Comrade.logdensityof(t.tpost, x), hv
end

# Device buffers are freed only when Julia collects their wrappers, which it does not do on
# device memory pressure.
collect_device_garbage(block) = block % 50 == 0 ? GC.gc() : (block % 5 == 0 && GC.gc(false))

"""
    sample_kinetic(out, post, xopt, strategy, rng; starts=nothing) -> Comrade.PosteriorSamples

`strategy.chains` chains of the KineticSamplers algorithm `strategy.algorithm` (MCLMC or
WALNUTS) on `post` in the latent space of `strategy.latent_space`, compiled with Reactant, started at `xopt`
jittered by `$(KINETIC_START_JITTER)` standard normals per latent coordinate, or chain `k` at
`starts[k]` when `starts` is given. A `PooledAdaptation` of `strategy.nadapt` MCLMC steps or
WALNUTS draws per chain, in blocks of `strategy.chunk_size` MCLMC steps or of KineticSamplers'
default length for WALNUTS, adapts one step size and one low-rank inverse mass with stiff directions for all chains; MCLMC warms up without step
halving. Its history and parameters go to `out/mclmc_warmup.jls` or `out/walnuts_warmup.jls`.
For MCLMC, $(MCLMC_CHECK_BLOCKS) blocks at the sampling parameters, whose energy statistics are
logged per chain, must then pass `KineticSamplers.check_energy`.

Sampling keeps `strategy.nsample` draws per chain, chain `k` written as a Comrade sample
directory `out/chain_k`: the last position of each block for MCLMC, every draw for WALNUTS
(`WALNUTS(; max_depth = strategy.max_tree_depth)`). The moves of `strategy` run on every chain
during the warmup's adaptation phase and sampling, every $(MCLMC_MOVE_BLOCKS) blocks for MCLMC
and every $(WALNUTS_MOVE_BLOCKS) blocks for WALNUTS; a chain that moved restarts at its new position. Returns about
$(KINETIC_RETURNED_DRAWS) draws, evenly thinned from each chain, as a draws × chains
`PosteriorSamples`; the chains on disk keep every draw (see [`load_kinetic_chains`](@ref)).
"""
function sample_kinetic(out, post, xopt, strategy::FittingStrategy, rng; starts = nothing)
    K = strategy.chains
    chain_dirs = [joinpath(out, "chain_$k") for k in 1:K]
    existing = filter(ispath, chain_dirs)
    isempty(existing) || error(
        "$(first(existing)) exists; $(strategy.algorithm) sampling does not resume, so move the earlier chains away"
    )
    post_cpu = @set post.admode = nothing
    space = latent_space(strategy)
    tpost = Comrade.maybe_transport(
        Comrade.prepare_device(post_cpu, Comrade.ComradeBase.ReactantEx()), space
    )
    tb = Comrade.CoordinateView(post_cpu, space).tbase
    d = dimension(tb)
    target = KineticSamplers.Target(LatentTarget(tpost, d); backend = ADTypes.AutoReactant())
    alg, warmup_alg = kinetic_samplers(strategy)
    name = String(nameof(typeof(alg)))
    nadapt, block = strategy.nadapt, strategy.chunk_size
    layout = block_layout(alg, block)
    strategy.nsample % layout.draws == 0 || error(
        "$(strategy.nsample) draws per chain are not a whole number of blocks of $(layout.draws)"
    )
    log_blocks = cld(KINETIC_LOG_DRAWS, layout.draws)
    mass = KineticSamplers.HistoryMassAdaptation(;
        max_rank = 100, max_wide = 80, stiff = KineticSamplers.StiffDirections(),
        vecs_eltype = Float32
    )
    adapt = KineticSamplers.PooledAdaptation(
        nadapt; mass, block = alg isa KineticSamplers.WALNUTS ? nothing : block
    )
    warmup_block = KineticSamplers.block_length(adapt, warmup_alg)
    Θ = if isnothing(starts)
        Comrade.inverse(tb, xopt) .+ KINETIC_START_JITTER .* randn(rng, d, K)
    else
        length(starts) == K ||
            error("$(length(starts)) starting points for sampler.chains = $K chains")
        reduce(hcat, [Comrade.inverse(tb, s) for s in starts])
    end
    host_rngs(n) = [Random.Xoshiro(rand(rng, UInt64)) for _ in 1:n]
    restart_rngs = host_rngs(K)

    moves = isempty(strategy.moves) ? nothing : build_moves(
            post_cpu, strategy.moves, xopt; space, output = joinpath(mkpath(out), "moves.jls")
        )
    function move_chains(chains, phase, step, total)
        Θh = Array(chains.state.θ)
        moved = false
        for k in axes(Θh, 2)
            x = Reactant.to_rarray(Θh[:, k])
            x′ = moves(x, tpost, phase, rng; step, total, report = false)
            if x′ !== x
                Θh[:, k] = Array(x′)
                moved = true
            end
        end
        GC.gc()
        moved || return chains
        return KineticSamplers.init_chains(
            restart_rngs, target, phase === :warmup ? warmup_alg : alg, Θh
        )
    end

    @info "$name: $K chains on $d latent coordinates, warmup $nadapt $(layout.unit) per chain in " *
        "blocks of $warmup_block, $(strategy.nsample) draws per chain; $(repr(alg))"
    t0 = time()
    function between_blocks(chains, info)
        collect_device_garbage(info.block)
        if !isnothing(moves) && info.phase === :adaptation && info.block % layout.move_blocks == 0
            chains = move_chains(chains, :warmup, info.steps, nadapt)
        end
        info.block % log_blocks == 0 && @info @sprintf(
            "%s warmup, %s block %d: %d %s per chain (%.0f s)",
            name, info.phase, info.block, info.steps, layout.unit, time() - t0
        )
        _flush_logs()
        return chains
    end
    chains, params, history = KineticSamplers.warmup(
        host_rngs(K), target, adapt, warmup_alg, Θ; between_blocks, check = false
    )
    _log_warmup(alg, params, history, d, time() - t0)
    serialize(
        joinpath(out, "$(lowercase(name))_warmup.jls"), (; history, params = _host_params(params))
    )

    params = Reactant.to_rarray(params; track_numbers = Number)
    rngs = KineticSamplers.backend_rngs(ADTypes.AutoReactant(), host_rngs(K))
    chains, trace = check_chains(target, rngs, alg, params, chains, block, d)
    writers = [ChainDrawWriter(dir) for dir in chain_dirs]
    health = nothing
    nblocks = strategy.nsample ÷ layout.draws
    t0 = time()
    for b in 1:nblocks
        tb0 = time()
        chains, trace = KineticSamplers.run_block(
            rngs, target, alg, params, chains, trace, block, layout.recorder
        )
        draws = block_draws(alg, chains, trace, time() - tb0)
        for (Θh, stats) in draws
            for (k, w) in pairs(writers)
                push_draw!(w, Comrade.transform(tb, Θh[:, k]); stats[k]...)
            end
            counts = [draw_health(alg, s) for s in stats]
            health = isnothing(health) ? counts : map((a, c) -> map(+, a, c), health, counts)
        end
        if !isnothing(moves) && b % layout.move_blocks == 0
            chains = move_chains(chains, :sampling, b * layout.draws, strategy.nsample)
        end
        collect_device_garbage(b)
        if b % log_blocks == 0 || b == nblocks
            @info @sprintf(
                "%s sampling %d/%d draws per chain: %s, log density %s (%.0f s)",
                name, b * layout.draws, strategy.nsample, _health_text(health),
                join((@sprintf("%.0f", s.logp) for s in last(last(draws))), " "), time() - t0
            )
            _flush_logs()
        end
    end
    foreach(flush_draws!, writers)
    isnothing(moves) || foreach(move_summary(moves)) do m
        @info "move $(m.name): warmup $(m.warmup.accepted)/$(m.warmup.proposed), " *
            "sampling $(m.sampling.accepted)/$(m.sampling.proposed) accepted" * _scale_text(m.τ)
    end
    stride = max(1, strategy.nsample ÷ cld(KINETIC_RETURNED_DRAWS, K))
    return load_kinetic_chains(out, 1:stride:strategy.nsample)
end

# The sampler of `strategy` and the one its warmup runs.
function kinetic_samplers(strategy::FittingStrategy)
    if strategy.algorithm == "mclmc"
        # Halving would let stored positions reach regions whose curvature caps the adapted step.
        return KineticSamplers.MCLMC(; strategy.max_step_halvings),
            KineticSamplers.MCLMC(; max_step_halvings = 0)
    elseif strategy.algorithm == "walnuts"
        alg = KineticSamplers.WALNUTS(; max_depth = strategy.max_tree_depth)
        return alg, alg
    end
    return error("sampler.algorithm = $(repr(strategy.algorithm)) is not a KineticSamplers algorithm")
end

# A block of `block` MCLMC steps keeps its last position; one of `block` WALNUTS draws keeps
# every draw.
block_layout(::KineticSamplers.MCLMC, block) = (;
    unit = "steps", draws = 1, move_blocks = MCLMC_MOVE_BLOCKS,
    recorder = KineticSamplers.NoPositions(),
)
block_layout(::KineticSamplers.WALNUTS, block) = (;
    unit = "draws", draws = block, move_blocks = WALNUTS_MOVE_BLOCKS, recorder = KineticSamplers.RecordPositions(),
)

# The draws of a sampling block that took `seconds`, as (positions `d × K`, statistics of each
# chain) pairs.
function block_draws(::KineticSamplers.MCLMC, chains, trace, seconds)
    Θh = Array(chains.state.θ)
    logp, grads = Array(trace.logp), Array(trace.num_grads)
    diverged, halvings = Array(trace.diverged), Array(trace.step_halvings)
    stats = map(axes(Θh, 2)) do k
        (;
            logp = logp[end, k], num_grads = sum(view(grads, :, k)), time = seconds / size(Θh, 2),
            divergent = count(view(diverged, :, k)), halved = count(>(0), view(halvings, :, k)),
        )
    end
    return [(Θh, stats)]
end

function block_draws(::KineticSamplers.WALNUTS, chains, trace, seconds)
    positions = Array(trace.positions)
    logp, grads, depth = Array(trace.logp), Array(trace.num_grads), Array(trace.depth)
    accept, failed = Array(trace.accept_stat), Array(trace.leaf_failed)
    per_draw = seconds / (size(positions, 2) * size(positions, 3))
    return map(axes(positions, 2)) do j
        stats = map(axes(positions, 3)) do k
            (;
                logp = logp[j, k], num_grads = grads[j, k], time = per_draw, tree_depth = depth[j, k],
                accept_stat = accept[j, k], leaf_failed = failed[j, k],
            )
        end
        (positions[:, j, :], stats)
    end
end

draw_health(::KineticSamplers.MCLMC, s) = (; divergent_steps = s.divergent, halved_steps = s.halved)
draw_health(alg::KineticSamplers.WALNUTS, s) = (;
    failed_draws = Int(s.leaf_failed), max_depth_draws = Int(s.tree_depth == alg.max_depth),
)

_health_text(health) = join(
    ("$(replace(String(k), "_" => " ")) $(join(getfield.(health, k), " "))" for k in keys(first(health))),
    ", "
)

"""
    load_kinetic_chains(out, nsamples=:) -> Comrade.PosteriorSamples

The draws of the chains `out/chain_1`, `out/chain_2`, … of an MCLMC or WALNUTS run (see
[`sample_kinetic`](@ref)) as a draws × chains `PosteriorSamples`, with `nsamples` selecting
draws of each chain.
"""
function load_kinetic_chains(out, nsamples = Base.Colon())
    per_chain = [Comrade.load_samples(dir, nsamples) for dir in _chain_dirs(out)]
    lengths = length.(per_chain)
    allequal(lengths) || error(
        "the chains in $out have different numbers of draws, $lengths; pass `nsamples` to select the same draws of each"
    )
    return Comrade.PosteriorSamples(
        hcat(Comrade.postsamples.(per_chain)...), hcat(Comrade.samplerstats.(per_chain)...);
        metadata = Dict(:sampler => :KineticSamplers, :chains => length(per_chain))
    )
end

function _chain_dirs(out)
    names = filter(n -> occursin(r"^chain_\d+$", n), readdir(out))
    isempty(names) && error("no chains (chain_1, chain_2, …) in $out")
    sort!(names; by = n -> parse(Int, n[7:end]))
    return [joinpath(out, n) for n in names]
end

check_chains(target, rngs, alg::KineticSamplers.MCLMC, params, chains, block, d) =
    check_mclmc(target, rngs, alg, params, chains, block, d)
check_chains(target, rngs, ::KineticSamplers.WALNUTS, params, chains, block, d) = (chains, nothing)

# Runs MCLMC_CHECK_BLOCKS blocks and applies KineticSamplers' end-of-warmup energy check to them,
# logging each chain's statistics first.
function check_mclmc(target, rngs, alg, params, chains, block, d)
    recorder = KineticSamplers.NoPositions()
    trace = nothing
    energies = NamedTuple[]
    for _ in 1:MCLMC_CHECK_BLOCKS
        chains, trace = KineticSamplers.run_block(
            rngs, target, alg, params, chains, trace, block, recorder
        )
        push!(
            energies, (;
                energy_error = Array(trace.energy_error), diverged = Array(trace.diverged),
                step_halvings = Array(trace.step_halvings),
            )
        )
    end
    for k in axes(first(energies).energy_error, 2)
        chain_k = [map(a -> a[:, k:k], e) for e in energies]
        st = KineticSamplers.energy_statistics(chain_k, d)
        @info @sprintf(
            "MCLMC check, chain %d: halved %.3g, divergent %.3g, energy error variance per dimension %.3g",
            k, st.halved, st.divergent, st.energy_var
        )
    end
    KineticSamplers.check_energy(energies, d)
    return chains, trace
end

_host_mass(m) = (; stds = Array(m.stds), vecs = Array(m.vecs), vals = Array(m.vals))
_host_params(p::KineticSamplers.WalnutsParameters) =
    (; p.step_size, inverse_mass = _host_mass(p.inverse_mass))
_host_params(p) = (; p.step_size, p.trajectory_length, inverse_mass = _host_mass(p.inverse_mass))

function _log_warmup(::KineticSamplers.MCLMC, params, history, d, seconds)
    c = history.curvature
    isnothing(c) || @info @sprintf(
        "MCLMC curvature cap: step size cap %.4g at λ_q %.4g (λ median %.4g, max %.4g over %d positions)",
        c.step_size_cap, c.curvature, c.median, c.max, c.positions
    )
    @info @sprintf(
        "MCLMC warmup done in %.0f s: step size %.4g, trajectory length %.4g = %.2f √d, inverse mass rank %d",
        seconds, params.step_size, params.trajectory_length,
        params.trajectory_length / sqrt(d), size(params.inverse_mass.vecs, 2)
    )
    return nothing
end

function _log_warmup(::KineticSamplers.WALNUTS, params, history, d, seconds)
    @info @sprintf(
        "WALNUTS warmup done in %.0f s: step size %.4g, inverse mass rank %d after %d refits",
        seconds, params.step_size, size(params.inverse_mass.vecs, 2), count(history.refit)
    )
    return nothing
end

"""
    ChainDrawWriter(dir)

Draws of one chain written to `dir` as a Comrade sample directory (`samples/output_scan_*.jls`
and `parameters.jls`), $(KINETIC_DRAWS_PER_FILE) draws per file, readable with
`Comrade.load_samples(dir)` while the chain runs.
"""
mutable struct ChainDrawWriter
    dir::String
    samples::Vector{Any}
    stats::Vector{NamedTuple}
    files::Int
    draws::Int
end

ChainDrawWriter(dir) = ChainDrawWriter(mkpath(dir), Any[], NamedTuple[], 0, 0)

function push_draw!(w::ChainDrawWriter, θ; stats...)
    push!(w.samples, θ)
    push!(w.stats, NamedTuple(stats))
    length(w.samples) == KINETIC_DRAWS_PER_FILE && flush_draws!(w)
    return w
end

function flush_draws!(w::ChainDrawWriter)
    isempty(w.samples) && return w
    w.files += 1
    w.draws += length(w.samples)
    ps = Comrade.PosteriorSamples(map(identity, w.samples), StructArray(map(identity, w.stats)))
    mkpath(joinpath(w.dir, "samples"))
    serialize(
        joinpath(w.dir, "samples", @sprintf("output_scan_%05d.jls", w.files)),
        (samples = Comrade.postsamples(ps), stats = Comrade.samplerstats(ps))
    )
    serialize(
        joinpath(w.dir, "parameters.jls"),
        (; params = Comrade.DiskOutput(abspath(w.dir), w.files, KINETIC_DRAWS_PER_FILE, w.draws))
    )
    empty!(w.samples)
    empty!(w.stats)
    return w
end
