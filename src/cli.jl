const CLI_USAGE = """
Image a VLBI observation, configured entirely from four TOML files.

The sampler (AdvancedHMC NUTS, Reactant NUTS, MCLMC or WALNUTS) is part of the fitting
TOML, since selecting Reactant also changes the optimizer path.

Usage: bbcomrade --image <path> --instrument <path> --data <path> --fitting <path>
            [--outpath <path>] [--restart]

Julia flags go before a `--` separator, e.g. `bbcomrade --threads=4 -- --image ...`.

Options (`--name value` or `--name=value`):

  --image <path>       image/sky-model TOML (grid, mean model, flux, random-field order)
  --instrument <path>  instrument-model TOML (gain/leakage scheme + per-parameter priors)
  --data <path>        data TOML (file/array/averaging/noise + flag table)
  --fitting <path>     fitting-strategy TOML (optimizer, tempering schedule, sampler, Reactant)
  --outpath <path>     output base path for the run (default `Runs/run`)

Flags:

  --restart            resume the run from a previously serialized optimum at `--outpath`
                       instead of re-optimizing. This is a one-off action, deliberately not
                       stored in any TOML.
  -h, --help           print this message
"""

const CLI_REQUIRED = ("image", "instrument", "data", "fitting")
const CLI_OPTIONS = (CLI_REQUIRED..., "outpath")

"""
    bbcomrade(args) -> Union{Nothing, Comrade.PosteriorSamples}

Run [`image_from_toml`](@ref) from command-line arguments, as the `bbcomrade` app does: the
four config paths are given with `--image`, `--instrument`, `--data` and `--fitting`, the
output base with `--outpath` (default `Runs/run`), and `--restart` resumes from the
serialized optimum there. `--help` prints the usage and returns `nothing`.
"""
function bbcomrade(args)
    parsed = _parse_cli(args)
    isnothing(parsed) && (print(CLI_USAGE); return nothing)
    opts, restart = parsed
    return image_from_toml(
        opts["image"], opts["instrument"], opts["data"], opts["fitting"];
        outpath = get(opts, "outpath", "Runs/run"), restart
    )
end

# The options and the restart flag, or `nothing` when help is asked for.
function _parse_cli(args)
    opts = Dict{String, String}()
    restart = false
    i = firstindex(args)
    while i <= lastindex(args)
        arg = args[i]
        if arg in ("-h", "--help")
            return nothing
        elseif arg == "--restart"
            restart = true
        elseif startswith(arg, "--")
            name, eq, value = _split_option(arg[3:end])
            name in CLI_OPTIONS || error("unknown option --$name\n\n$CLI_USAGE")
            if !eq
                i < lastindex(args) || error("option --$name needs a value")
                i += 1
                value = args[i]
            end
            haskey(opts, name) && error("option --$name given twice")
            opts[name] = value
        else
            error("unexpected argument '$arg'\n\n$CLI_USAGE")
        end
        i += 1
    end
    missing_opts = [n for n in CLI_REQUIRED if !haskey(opts, n)]
    isempty(missing_opts) ||
        error("missing required option(s): " * join("--" .* missing_opts, ", ") * "\n\n$CLI_USAGE")
    return opts, restart
end

# Split `name=value` into (name, true, value); a bare `name` gives (name, false, "").
function _split_option(s)
    k = findfirst('=', s)
    isnothing(k) && return s, false, ""
    return s[1:prevind(s, k)], true, s[nextind(s, k):end]
end

# The entry point of the `bbcomrade` app (`julia -m BlackboxComrade`).
function (@main)(args)
    bbcomrade(args)
    return 0
end
