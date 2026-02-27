#!/usr/bin/env julia
#=
Build aggregated CSVs combining solvers per data type.
- Default: Boscia + SCIPSDP_oa + SCIPSDP_bnb → independent/correlated_by_dimension/N_construction (4 files).
- Smoothing mode (--smoothing): 4 Boscia smoothing regimes (large_mu, small_mu, decay_0.9, decay_0.7) →
  smoothing_independent/correlated_by_dimension/N_construction (4 files).
- AGC mode (--agc): Boscia + SCIPSDP_oa + SCIPSDP_bnb for AGC setups →
  agc_correlated_connected_by_dimension.csv and agc_independent_disconnected_by_dimension.csv.
- Metrics: geometric mean of time, std w.r.t. geom mean, n_solved, pct_solved, rel_gap geom (unsolved),
  failed_instances, avg_lmo_calls, avg_nodes, etc.

Run after merge_single_runs_to_csv.jl. Reads merged CSVs from csv/Boscia and csv/SCIPSDP.
=#

using CSV, DataFrames, Statistics

const CSV_BASE = joinpath(@__DIR__, "csv")
const BOSCIA_DIR = joinpath(CSV_BASE, "Boscia")
const SCIPSDP_DIR = joinpath(CSV_BASE, "SCIPSDP")
const TIME_LIMIT = 3600
const BOSCIA_DELIM = ';'
const SCIPSDP_DELIM = ','

const SCIPSOLVERS = ["Boscia", "SCIPSDP_oa", "SCIPSDP_bnb"]

# Solved = reached optimal (or gap limit) within time
function is_solved(termination, time)
    t = termination isa String ? termination : string(termination)
    return t in ("OPTIMAL", "GAPLIMIT", "OPTIMALITY_PROVED") && (time isa Number && time < TIME_LIMIT)
end

# Placeholder row from merge script (missing run)
function is_failed(termination, solution, solution_source)
    t = termination isa String ? termination : string(termination)
    if t != "ERROR"
        return false
    end
    if solution_source !== missing && solution_source !== nothing && solution_source == "missing"
        return true
    end
    return solution isa Number && !isfinite(solution) || solution >= 1e30
end

# Geometric mean; skip non-finite and non-positive
function geom_mean(v)
    x = [e for e in v if isfinite(e) && e isa Number && e > 0]
    isempty(x) && return missing
    return exp(sum(log, x) / length(x))
end

# N construction label from (n, N): same formulas as merge_single_runs_to_csv / run_optimal_design
function n_construction_label(n_param, N_val)
    n = n_param isa Integer ? Int(n_param) : floor(Int, n_param)
    n_rd = floor(Int, 3n / 4)
    n_one = floor(Int, 1.5 * n)
    n_log = floor(Int, 1.5 * n * log(n))
    N = N_val isa Integer ? Int(N_val) : floor(Int, N_val)
    N == n_rd && return "rank_deficient"
    N == n_one && return "one"
    N == n_log && return "log"
    return "other"
end

# Std of values w.r.t. geometric mean (arithmetic std around geom mean)
function std_wrt_geom_mean(times)
    g = geom_mean(times)
    g === missing && return missing
    valid = [t for t in times if isfinite(t) && t isa Number && t > 0]
    isempty(valid) && return missing
    return sqrt(sum((t - g)^2 for t in valid) / length(valid))
end

function load_and_normalize_boscia(corr::Bool)
    type = corr ? "correlated" : "independent"
    path = joinpath(BOSCIA_DIR, "boscia_E_optimality_$(type)_cont_merged.csv")
    isfile(path) || return nothing
    df = CSV.read(path, DataFrame; delim=BOSCIA_DELIM, silencewarnings=true)
    n = nrow(df)
    df[!, :solver] = fill("Boscia", n)
    df[!, :dimension] = df.numberOfExperiments
    df[!, :rel_gap] = coalesce.(df.rel_dual_gap, Inf)
    df[!, :solved] = [is_solved(row.termination, row.time) for row in eachrow(df)]
    df[!, :failed] = [is_failed(row.termination, row.solution, get(row, :solution_source, missing)) for row in eachrow(df)]
    df[!, :N_construction] = [n_construction_label(row.numberOfParameters, row.N) for row in eachrow(df)]
    # Unified stats for aggregation (Boscia: ncalls, num_nodes; others 0)
    df[!, :nodes] = df.num_nodes
    df[!, :ncalls] = hasproperty(df, :ncalls) ? df.ncalls : fill(0, n)
    df[!, :n_cuts_applied] = fill(0, n)
    df[!, :n_sdp_iters] = fill(0, n)
    return df
end

const SMOOTHING_REGIMES = ["large_mu", "small_mu", "decay_0.9", "decay_0.7"]

function load_and_normalize_boscia_smoothing(regime::String, corr::Bool)
    type = corr ? "correlated" : "independent"
    path = joinpath(BOSCIA_DIR, "boscia_smoothing_$(regime)_E_optimality_$(type)_merged.csv")
    isfile(path) || return nothing
    df = CSV.read(path, DataFrame; delim=BOSCIA_DELIM, silencewarnings=true)
    n = nrow(df)
    df[!, :solver] = fill(regime, n)
    df[!, :dimension] = df.numberOfExperiments
    df[!, :rel_gap] = hasproperty(df, :rel_dual_gap) ? coalesce.(df.rel_dual_gap, Inf) : fill(Inf, n)
    df[!, :solved] = [is_solved(row.termination, row.time) for row in eachrow(df)]
    df[!, :failed] = [is_failed(row.termination, row.solution, get(row, :solution_source, missing)) for row in eachrow(df)]
    df[!, :N_construction] = [n_construction_label(row.numberOfParameters, row.N) for row in eachrow(df)]
    df[!, :nodes] = hasproperty(df, :num_nodes) ? df.num_nodes : fill(0, n)
    df[!, :ncalls] = hasproperty(df, :ncalls) ? df.ncalls : fill(0, n)
    df[!, :n_cuts_applied] = fill(0, n)
    df[!, :n_sdp_iters] = fill(0, n)
    return df
end

function load_and_normalize_scipsdp(mode::String, corr::Bool)
    type = corr ? "correlated" : "independent"
    path = joinpath(SCIPSDP_DIR, "scip_sdp_$(mode)_E_optimality_$(type)_cont_merged.csv")
    isfile(path) || return nothing
    df = CSV.read(path, DataFrame; delim=SCIPSDP_DELIM, silencewarnings=true)
    n = nrow(df)
    df[!, :solver] = fill("SCIPSDP_$(mode)", n)
    df[!, :dimension] = df.numberOfExperiments
    if !hasproperty(df, :rel_gap)
        df[!, :rel_gap] = fill(Inf, n)
    end
    df[!, :solved] = [is_solved(row.termination, row.time) for row in eachrow(df)]
    df[!, :failed] = [is_failed(row.termination, row.solution, get(row, :solution_source, missing)) for row in eachrow(df)]
    df[!, :N_construction] = [n_construction_label(row.numberOfParameters, row.N) for row in eachrow(df)]
    # Unified stats: nodes (all), ncalls=0, n_cuts_applied (oa only), n_sdp_iters (bnb only)
    df[!, :nodes] = hasproperty(df, :n_nodes) ? df.n_nodes : fill(0, n)
    df[!, :ncalls] = fill(0, n)
    df[!, :n_cuts_applied] = (mode == "oa" && hasproperty(df, :n_cuts_applied)) ? df.n_cuts_applied : fill(0, n)
    df[!, :n_sdp_iters] = (mode == "bnb" && hasproperty(df, :n_sdp_iters)) ? coalesce.(df.n_sdp_iters, 0) : fill(0, n)
    return df
end

function combined_table(corr::Bool)
    dfs = DataFrame[]
    for (label, loader) in [
        ("Boscia", () -> load_and_normalize_boscia(corr)),
        ("SCIPSDP_oa", () -> load_and_normalize_scipsdp("oa", corr)),
        ("SCIPSDP_bnb", () -> load_and_normalize_scipsdp("bnb", corr)),
    ]
        df = loader()
        df === nothing && continue
        push!(dfs, df)
    end
    isempty(dfs) && return nothing
    common = [:seed, :dimension, :N, :numberOfParameters, :N_construction, :time, :solution, :termination, :solver, :rel_gap, :solved, :failed, :nodes, :ncalls, :n_cuts_applied, :n_sdp_iters]
    available = [n for n in common if all(hasproperty(d, n) for d in dfs)]
    return vcat([df[:, available] for df in dfs]...)
end

function combined_table_smoothing(corr::Bool)
    dfs = DataFrame[]
    for regime in SMOOTHING_REGIMES
        df = load_and_normalize_boscia_smoothing(regime, corr)
        df === nothing && continue
        push!(dfs, df)
    end
    isempty(dfs) && return nothing
    common = [:seed, :dimension, :N, :numberOfParameters, :N_construction, :time, :solution, :termination, :solver, :rel_gap, :solved, :failed, :nodes, :ncalls, :n_cuts_applied, :n_sdp_iters]
    available = [n for n in common if all(hasproperty(d, n) for d in dfs)]
    return vcat([df[:, available] for df in dfs]...)
end

function load_and_normalize_boscia_agc(corr::Bool, connected::Bool)
    type = corr ? "correlated" : "independent"
    conn = connected ? "connected" : "disconnected"
    path = joinpath(BOSCIA_DIR, "boscia_AGC_optimality_$(type)_$(conn)_merged.csv")
    isfile(path) || return nothing
    df = CSV.read(path, DataFrame; delim=BOSCIA_DELIM, silencewarnings=true)
    n = nrow(df)
    df[!, :solver] = fill("Boscia", n)
    df[!, :dimension] = df.numberOfExperiments
    df[!, :rel_gap] = hasproperty(df, :rel_dual_gap) ? coalesce.(df.rel_dual_gap, Inf) : fill(Inf, n)
    df[!, :solved] = [is_solved(row.termination, row.time) for row in eachrow(df)]
    df[!, :failed] = [is_failed(row.termination, row.solution, get(row, :solution_source, missing)) for row in eachrow(df)]
    # For AGC, N_construction is not meaningful; mark as "other"
    df[!, :N_construction] = fill("other", n)
    df[!, :nodes] = hasproperty(df, :num_nodes) ? df.num_nodes : fill(0, n)
    df[!, :ncalls] = hasproperty(df, :ncalls) ? df.ncalls : fill(0, n)
    df[!, :n_cuts_applied] = fill(0, n)
    df[!, :n_sdp_iters] = fill(0, n)
    return df
end

function load_and_normalize_scipsdp_agc(mode::String, corr::Bool, connected::Bool)
    type = corr ? "correlated" : "independent"
    conn = connected ? "connected" : "disconnected"
    path = joinpath(SCIPSDP_DIR, "scip_sdp_$(mode)_AGC_optimality_$(type)_$(conn)_merged.csv")
    isfile(path) || return nothing
    df = CSV.read(path, DataFrame; delim=SCIPSDP_DELIM, silencewarnings=true)
    n = nrow(df)
    df[!, :solver] = fill("SCIPSDP_$(mode)", n)
    df[!, :dimension] = df.numberOfExperiments
    df[!, :rel_gap] = hasproperty(df, :rel_gap) ? df.rel_gap : fill(Inf, n)
    df[!, :solved] = [is_solved(row.termination, row.time) for row in eachrow(df)]
    df[!, :failed] = [is_failed(row.termination, row.solution, get(row, :solution_source, missing)) for row in eachrow(df)]
    df[!, :N_construction] = fill("other", n)
    df[!, :nodes] = hasproperty(df, :n_nodes) ? df.n_nodes : fill(0, n)
    df[!, :ncalls] = fill(0, n)
    df[!, :n_cuts_applied] = (mode == "oa" && hasproperty(df, :n_cuts_applied)) ? df.n_cuts_applied : fill(0, n)
    df[!, :n_sdp_iters] = (mode == "bnb" && hasproperty(df, :n_sdp_iters)) ? coalesce.(df.n_sdp_iters, 0) : fill(0, n)
    return df
end

function combined_table_agc(corr::Bool, connected::Bool)
    dfs = DataFrame[]
    for solver in SCIPSOLVERS
        df = if solver == "Boscia"
            load_and_normalize_boscia_agc(corr, connected)
        elseif solver == "SCIPSDP_oa"
            load_and_normalize_scipsdp_agc("oa", corr, connected)
        else
            load_and_normalize_scipsdp_agc("bnb", corr, connected)
        end
        df === nothing && continue
        push!(dfs, df)
    end
    isempty(dfs) && return nothing
    common = [:seed, :dimension, :N, :numberOfParameters, :N_construction, :time, :solution, :termination, :solver, :rel_gap, :solved, :failed, :nodes, :ncalls, :n_cuts_applied, :n_sdp_iters]
    available = [n for n in common if all(hasproperty(d, n) for d in dfs)]
    return vcat([df[:, available] for df in dfs]...)
end

function aggregate_by(df::DataFrame, group_col::Symbol)
    g = groupby(df, [:solver, group_col])
    rows = []
    for sdf in g
        solver = first(sdf.solver)
        grp = first(sdf[!, group_col])
        times = sdf.time
        rel_gaps = sdf.rel_gap
        solved = sdf.solved
        failed = sdf.failed
        n = nrow(sdf)
        n_solved = count(solved)
        pct_solved = n > 0 ? 100.0 * n_solved / n : 0.0
        time_geom = geom_mean(times)
        time_std_geom = std_wrt_geom_mean(times)
        unsolved_rel = [r for (r, s) in zip(rel_gaps, solved) if !s && isfinite(r) && r isa Number && r > 0]
        rel_gap_geom_unsolved = isempty(unsolved_rel) ? missing : exp(sum(log, unsolved_rel) / length(unsolved_rel))
        n_failed = count(failed)
        # Averages over solved instances only; 0 when not applicable for that solver
        solved_idx = findall(solved)
        n_sol = length(solved_idx)
        is_boscia_like = solver == "Boscia" || solver in SMOOTHING_REGIMES
        avg_lmo_calls = (is_boscia_like && n_sol > 0) ? round(sum(sdf.ncalls[solved_idx]) / n_sol; digits=2) : 0.0
        avg_nodes = n_sol > 0 ? round(sum(sdf.nodes[solved_idx]) / n_sol; digits=2) : 0.0
        avg_cuts = (solver == "SCIPSDP_oa" && n_sol > 0) ? round(sum(skipmissing(sdf.n_cuts_applied[solved_idx])) / n_sol; digits=2) : 0.0
        avg_sdp_iters = (solver == "SCIPSDP_bnb" && n_sol > 0) ? round(sum(coalesce.(sdf.n_sdp_iters[solved_idx], 0)) / n_sol; digits=2) : 0.0
        row_dict = Dict(
            :solver => solver,
            :n_instances => n,
            :n_solved => n_solved,
            :pct_solved => round(pct_solved; digits=2),
            :time_geom_mean => time_geom === missing ? missing : round(time_geom; digits=4),
            :time_std_wrt_geom => time_std_geom === missing ? missing : round(time_std_geom; digits=4),
            :rel_gap_geom_mean_unsolved => rel_gap_geom_unsolved === missing ? missing : round(rel_gap_geom_unsolved; digits=6),
            :failed_instances => n_failed,
            :avg_lmo_calls => avg_lmo_calls,
            :avg_nodes => avg_nodes,
            :avg_cuts => avg_cuts,
            :avg_sdp_iters => avg_sdp_iters,
        )
        row_dict[group_col] = grp
        push!(rows, row_dict)
    end
    out = DataFrame(rows)
    # Consistent column order: group, solver, then metrics (incl. solver-specific avgs over solved only)
    order = [group_col, :solver, :n_instances, :n_solved, :pct_solved, :time_geom_mean, :time_std_wrt_geom, :rel_gap_geom_mean_unsolved, :failed_instances, :avg_lmo_calls, :avg_nodes, :avg_cuts, :avg_sdp_iters]
    nms = names(out)
    # Match both Symbol and String column names (DataFrame(rows) may use either)
    cols = [c for c in order if c in nms || string(c) in nms]
    if isempty(cols)
        return out
    end
    idx = [c in nms ? c : string(c) for c in order if c in nms || string(c) in nms]
    return out[:, idx]
end

function run_aggregation(; out_dir=nothing, smoothing=false, verbose=true)
    out_dir = something(out_dir, joinpath(CSV_BASE, "aggregated"))
    mkpath(out_dir)
    prefix = smoothing ? "smoothing_" : ""
    if verbose
        println(smoothing ? "Aggregating Boscia smoothing regimes (4) by data type and dimension/N." : "Aggregating merged results (Boscia + SCIPSDP) by data type and dimension/N.")
        println("Output directory: $out_dir")
    end
    for corr in (false, true)
        data_type = corr ? "correlated" : "independent"
        df = smoothing ? combined_table_smoothing(corr) : combined_table(corr)
        if df === nothing
            verbose && println("No data for $(prefix)$data_type, skipping.")
            continue
        end
        if verbose
            println("\n--- $(prefix)$data_type ---")
            println("  Combined rows: $(nrow(df)), solvers/regimes: $(unique(df.solver))")
        end
        by_dim = aggregate_by(df, :dimension)
        by_n = aggregate_by(df, :N_construction)
        n_order = ["rank_deficient", "one", "log"]
        col_n = :N_construction in names(by_n) ? :N_construction : "N_construction"
        perm = sortperm(by_n[!, col_n]; by=x -> (idx = findfirst(==(string(x)), n_order); idx === nothing ? 4 : idx))
        by_n = by_n[perm, :]
        out_dim = joinpath(out_dir, "$(prefix)$(data_type)_by_dimension.csv")
        out_n = joinpath(out_dir, "$(prefix)$(data_type)_by_N_construction.csv")
        CSV.write(out_dim, by_dim)
        CSV.write(out_n, by_n)
        if verbose
            println("  Wrote $out_dim ($(nrow(by_dim)) rows)")
            println("  Wrote $out_n ($(nrow(by_n)) rows)")
        end
    end
    if verbose
        println("\nDone. Outputs: $(prefix)*_by_dimension.csv, $(prefix)*_by_N_construction.csv")
    end
end

function run_aggregation_agc(; out_dir=nothing, verbose=true)
    out_dir = something(out_dir, joinpath(CSV_BASE, "aggregated"))
    mkpath(out_dir)
    if verbose
        println("Aggregating AGC results (Boscia + SCIPSDP) by dimension.")
        println("Output directory: $out_dir")
    end
    # Only two AGC setups are used: correlated_connected and independent_disconnected
    setups = [
        (true,  true,  "correlated_connected"),
        (false, false, "independent_disconnected"),
    ]
    for (corr, connected, tag) in setups
        df = combined_table_agc(corr, connected)
        if df === nothing
            verbose && println("No AGC data for $tag, skipping.")
            continue
        end
        if verbose
            println("\n--- agc_$tag ---")
            println("  Combined rows: $(nrow(df)), solvers: $(unique(df.solver))")
        end
        by_dim = aggregate_by(df, :dimension)
        out_dim = joinpath(out_dir, "agc_$(tag)_by_dimension.csv")
        CSV.write(out_dim, by_dim)
        if verbose
            println("  Wrote $out_dim ($(nrow(by_dim)) rows)")
        end
    end
    if verbose
        println("\nDone. Outputs: agc_*_by_dimension.csv")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    smoothing = "--smoothing" in ARGS
    agc = "--agc" in ARGS
    if smoothing
        filter!(x -> x != "--smoothing", ARGS)
    end
    if agc
        filter!(x -> x != "--agc", ARGS)
        run_aggregation_agc()
    else
        run_aggregation(; smoothing)
    end
end
