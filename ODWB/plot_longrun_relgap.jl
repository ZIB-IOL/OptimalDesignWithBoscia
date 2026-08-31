#!/usr/bin/env julia
#=
Relative-gap plots for long E-optimal runs (m = 500 … 10000).

Compares Boscia (optimized) vs SCIPSDP OA / BnB.
For each (data type × N construction), generates:
- a dimension plot with a min–max band across seeds around the median gap;
- a cumulative gap plot showing the percentage of all intended instances
  solved to a given relative gap or better within the time limit.

Usage:
  julia --project=. plot_longrun_relgap.jl
=#

using CSV
using DataFrames
using Plots
using Statistics
using Plots.PlotMeasures: mm

include(joinpath(@__DIR__, "colours.jl"))
include(joinpath(@__DIR__, "..", "plot", "plot_style.jl"))

const TIME_LIMIT = 3600.0
const LONG_DIMS = [500, 1000, 2000, 5000, 8000, 10000]
const SEEDS = 1:3
const EXPECTED_INSTANCES_PER_PANEL = length(LONG_DIMS) * length(SEEDS)
const BOSCIA_DIR = joinpath(@__DIR__, "csv", "Boscia")
const SCIPSDP_DIR = joinpath(@__DIR__, "csv", "SCIPSDP")
const OUT_DIRS = [
    joinpath(@__DIR__, "plots"),
    "/Users/deborah/Documents/research_projects/Smoothing-in-Boscia/paper",
]

rgb(t) = RGB(t[1], t[2], t[3])
const SOLVER_STYLE = Dict(
    "Boscia" => (color=rgb(cb_green_lime), label="Boscia"),
    "SCIPSDP_oa" => (color=rgb(cb_salmon_pink), label="SCIPSDP (OA)"),
    "SCIPSDP_bnb" => (color=rgb(cb_green_sea), label="SCIPSDP (B&B)"),
)
const SOLVER_ORDER = ["Boscia", "SCIPSDP_oa", "SCIPSDP_bnb"]

"""Long-run E constructions: N = ⌊1.5 n⌋ (\"one\") and N = ⌊1.5 n log n⌋ (\"log\")."""
function long_run_n_construction(n::Integer, N::Integer)
    n_one = floor(Int, 1.5 * n)
    n_log = floor(Int, 1.5 * n * log(n))
    N == n_one && return "one"
    N == n_log && return "log"
    return "other"
end

function parse_float(x; default=NaN)
    (x === missing || x === nothing) && return default
    v = tryparse(Float64, string(x))
    return v === nothing ? default : v
end

function read_single_csv(path::String; delim)
    isfile(path) || return nothing
    df = CSV.read(path, DataFrame; delim=delim, silencewarnings=true)
    nrow(df) == 0 && return nothing
    return df[end, :]  # last row = final status
end

function boscia_path(dtype::String, m, n, N, seed)
    return joinpath(BOSCIA_DIR, "boscia_optimized_E_optimality_$(dtype)__$(m)_$(n)_$(N)_$(seed).csv")
end

function scipsdp_path(mode::String, dtype::String, m, n, N, seed)
    # Prefer plain singles; fall back to *_cont_* if present.
    plain = joinpath(SCIPSDP_DIR, "scip_sdp_$(mode)_E_optimality_$(dtype)__$(m)_$(n)_$(N)_$(seed).csv")
    cont = joinpath(SCIPSDP_DIR, "scip_sdp_$(mode)_E_optimality_$(dtype)_cont__$(m)_$(n)_$(N)_$(seed).csv")
    isfile(plain) && return plain
    isfile(cont) && return cont
    return plain
end

function load_boscia_row(dtype, m, n, N, seed)
    row = read_single_csv(boscia_path(dtype, m, n, N, seed); delim=';')
    row === nothing && return nothing
    return (
        solver="Boscia",
        dimension=Int(m),
        n=Int(n),
        N=Int(N),
        seed=Int(seed),
        rel_gap=parse_float(row.rel_dual_gap),
        termination=string(row.termination),
        feasible=true,
    )
end

function load_scipsdp_row(mode::String, dtype, m, n, N, seed)
    row = read_single_csv(scipsdp_path(mode, dtype, m, n, N, seed); delim=',')
    row === nothing && return nothing
    feasible = true
    if hasproperty(row, :feasible)
        f = row.feasible
        if f isa Bool
            feasible = f
        elseif f isa Number
            feasible = f != 0
        else
            feasible = !(lowercase(strip(string(f))) in ("false", "0", "no", "f"))
        end
    end
    return (
        solver="SCIPSDP_$(mode)",
        dimension=Int(m),
        n=Int(n),
        N=Int(N),
        seed=Int(seed),
        rel_gap=parse_float(row.rel_gap),
        termination=string(row.termination),
        feasible=feasible,
    )
end

function collect_longrun_rows(dtype::String)
    rows = NamedTuple[]
    # Discover (n,N) pairs from Boscia optimized filenames for this dtype.
    rx = Regex("^boscia_optimized_E_optimality_$(dtype)__(\\d+)_(\\d+)_(\\d+)_(\\d+)\\.csv\$")
    pairs = Dict{Int,Vector{Tuple{Int,Int}}}()  # m => unique (n,N)
    for fname in readdir(BOSCIA_DIR)
        mobj = match(rx, fname)
        mobj === nothing && continue
        m, n, N, _ = parse.(Int, mobj.captures)
        m in LONG_DIMS || continue
        push!(get!(pairs, m, Tuple{Int,Int}[]), (n, N))
    end
    for m in LONG_DIMS
        haskey(pairs, m) || continue
        for (n, N) in unique(pairs[m])
            for seed in SEEDS
                b = load_boscia_row(dtype, m, n, N, seed)
                b !== nothing && push!(rows, b)
                oa = load_scipsdp_row("oa", dtype, m, n, N, seed)
                oa !== nothing && push!(rows, oa)
                bnb = load_scipsdp_row("bnb", dtype, m, n, N, seed)
                bnb !== nothing && push!(rows, bnb)
            end
        end
    end
    return rows
end

function rows_to_df(rows)
    isempty(rows) && return DataFrame()
    df = DataFrame(rows)
    df[!, :N_construction] = [long_run_n_construction(r.n, r.N) for r in eachrow(df)]
    # Keep invalid rows so cumulative percentages use the full instance-set
    # denominator. Invalid gaps are ignored in the numerator.
    gaps = Float64[]
    for g in df.rel_gap
        if g isa Number && isfinite(g)
            push!(gaps, max(Float64(g), 0.0))
        else
            push!(gaps, NaN)
        end
    end
    df[!, :rel_gap] = gaps
    return df
end

"""Median / min / max of gaps at each dimension (across seeds)."""
function band_stats(sub::DataFrame)
    dims = sort(unique(Int.(sub.dimension)))
    xs = Int[]
    med = Float64[]
    lo = Float64[]
    hi = Float64[]
    for m in dims
        vals = Float64[v for v in sub[sub.dimension .== m, :rel_gap] if isfinite(v)]
        isempty(vals) && continue
        push!(xs, m)
        push!(med, median(vals))
        push!(lo, minimum(vals))
        push!(hi, maximum(vals))
    end
    return xs, med, lo, hi
end

function make_plot(df::DataFrame, dtype::String, ncons::String; out_bases)
    sub = df[(df.N_construction .== ncons) .& (df.feasible .== true), :]
    nrow(sub) == 0 && return false

    plt = plot(
        xlabel="Dimension m",
        ylabel="Relative gap",
        legend=:topleft,
        grid=true,
        framestyle=:box,
        size=(640, 380),
        title="";
        cm_plot_kwargs(guidefontsize=11, tickfontsize=10, legendfontsize=9, left_margin=3mm, bottom_margin=3mm, right_margin=3mm, top_margin=3mm)...
    )

    any_series = false
    for solver in SOLVER_ORDER
        ssub = sub[sub.solver .== solver, :]
        nrow(ssub) == 0 && continue
        xs, med, lo, hi = band_stats(ssub)
        isempty(xs) && continue
        style = SOLVER_STYLE[solver]
        ribbon_lo = med .- lo
        ribbon_hi = hi .- med
        plot!(
            plt,
            xs,
            med;
            ribbon=(ribbon_lo, ribbon_hi),
            fillalpha=0.22,
            color=style.color,
            label=style.label,
            linewidth=2.2,
            marker=:circle,
            markersize=4,
        )
        any_series = true
    end
    any_series || return false

    fname = "E_longrun_relgap_$(dtype)_$(ncons).pdf"
    for out_dir in out_bases
        mkpath(out_dir)
        out = joinpath(out_dir, fname)
        savefig(plt, out)
        println("Wrote $out")
    end
    return true
end

"""Empirical cumulative gap points, using `total` as the full instance denominator."""
function cumulative_gap_points(sub::DataFrame, total::Int, max_gap::Float64)
    deduplicated = unique(sub, [:dimension, :seed])
    gaps = sort(Float64[g for g in deduplicated.rel_gap if isfinite(g) && g >= 0])
    isempty(gaps) && return Float64[], Float64[]

    thresholds = unique(gaps)
    xs = Float64[]
    ys = Float64[]
    if first(thresholds) > 0
        push!(xs, 0.0)
        push!(ys, 0.0)
    end
    for threshold in thresholds
        push!(xs, threshold)
        push!(ys, 100 * searchsortedlast(gaps, threshold) / total)
    end
    if last(xs) < max_gap
        push!(xs, max_gap)
        push!(ys, last(ys))
    end
    return xs, ys
end

function make_cumulative_plot(df::DataFrame, dtype::String, ncons::String; out_bases)
    sub = df[df.N_construction .== ncons, :]
    nrow(sub) == 0 && return false

    valid_gaps = Float64[
        r.rel_gap for r in eachrow(sub)
        if r.feasible && isfinite(r.rel_gap) && r.rel_gap >= 0
    ]
    isempty(valid_gaps) && return false
    max_gap = max(maximum(valid_gaps), eps(Float64))

    plt = plot(
        xlabel="Relative gap",
        ylabel="% instances",
        legend=:bottomright,
        grid=true,
        framestyle=:box,
        size=(640, 380),
        xlims=(0, max_gap),
        ylims=(0, 100),
        yticks=0:10:100,
        title="";
        cm_plot_kwargs(guidefontsize=11, tickfontsize=10, legendfontsize=9, left_margin=3mm, bottom_margin=3mm, right_margin=3mm, top_margin=3mm)...
    )

    any_series = false
    for solver in SOLVER_ORDER
        solver_rows = sub[(sub.solver .== solver) .& (sub.feasible .== true), :]
        xs, ys = cumulative_gap_points(
            solver_rows,
            EXPECTED_INSTANCES_PER_PANEL,
            max_gap,
        )
        isempty(xs) && continue
        style = SOLVER_STYLE[solver]
        plot!(
            plt,
            xs,
            ys;
            seriestype=:steppost,
            color=style.color,
            label=style.label,
            linewidth=2.2,
            marker=:circle,
            markersize=3.5,
        )
        any_series = true
    end
    any_series || return false

    fname = "E_longrun_relgap_cdf_$(dtype)_$(ncons).pdf"
    for out_dir in out_bases
        mkpath(out_dir)
        out = joinpath(out_dir, fname)
        savefig(plt, out)
        println("Wrote $out")
    end
    return true
end

function main()
    mkpath.(OUT_DIRS)
    n_plots = 0
    for dtype in ("correlated", "independent")
        println("--- $dtype ---")
        rows = collect_longrun_rows(dtype)
        df = rows_to_df(rows)
        println("  loaded $(nrow(df)) rows; constructions=$(sort(unique(String.(df.N_construction))))")
        for ncons in ("one", "log")
            n_plots += make_plot(df, dtype, ncons; out_bases=OUT_DIRS) ? 1 : 0
            n_plots += make_cumulative_plot(df, dtype, ncons; out_bases=OUT_DIRS) ? 1 : 0
        end
    end
    println("Generated $n_plots long-run relative-gap plots.")
end

main()
