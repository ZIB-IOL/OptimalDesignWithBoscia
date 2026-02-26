#!/usr/bin/env julia
#=
Merge single-run CSVs into one CSV per group (solver + mode + data type + correlated/independent).
- Keeps all single-run CSV files unchanged.
- E-optimal / SCIP: 75 runs per group (dimensions 50, 80, 100, 120, 150 × 5 seeds × 3 N values).
- Boscia smoothing: 36 runs per group (dimensions 50, 100, 150, 200 × 3 seeds × 3 N values);
  four regimes: large_mu (decay=1, μ_min<0.001), small_mu (decay=1, μ_min≥0.001), decay_0.9, decay_0.7.
- Missing instances get a placeholder row: time=3600, solution/scaled_solution=Inf, termination=ERROR,
  statistics = 0. Prints which instances are missing per group.

Usage: julia merge_single_runs_to_csv.jl [Boscia] [SCIPSDP] [BosciaSmoothing]
  No args = process all (Boscia, SCIPSDP, BosciaSmoothing).
=#

using CSV, DataFrames

const CSV_BASE = joinpath(@__DIR__, "csv")
const TIME_LIMIT = 3600
const SEEDS = 1:5
const DIMENSIONS_M = [50, 80, 100, 120, 150]
# Smoothing experiment: 4 dims × 3 seeds × 3 N = 36
const DIMENSIONS_SMOOTHING = [50, 100, 150, 200]
const SEEDS_SMOOTHING = 1:3

# (m) -> (n, N_list); E-optimal n = floor(sqrt(m)), three N values: rank_deficient, one, log
function expected_n_and_n_values(m::Int)
    n = floor(Int, sqrt(m))
    n_rd = floor(Int, 3n / 4)
    n_one = floor(Int, 1.5 * n)
    n_log = floor(Int, 1.5 * n * log(n))
    N_list = sort(unique([n_rd, n_one, n_log]))
    return n, N_list
end

function all_instance_keys()
    keys_list = Tuple{Int,Int,Int,Int}[]  # (m, n, N, seed)
    for m in DIMENSIONS_M
        n, N_list = expected_n_and_n_values(m)
        for N in N_list
            for seed in SEEDS
                push!(keys_list, (m, n, N, seed))
            end
        end
    end
    @assert length(keys_list) == 75 "expected 75 instances, got $(length(keys_list))"
    return keys_list
end

function all_instance_keys_smoothing()
    keys_list = Tuple{Int,Int,Int,Int}[]
    for m in DIMENSIONS_SMOOTHING
        n, N_list = expected_n_and_n_values(m)
        for N in N_list
            for seed in SEEDS_SMOOTHING
                push!(keys_list, (m, n, N, seed))
            end
        end
    end
    @assert length(keys_list) == 36 "expected 36 smoothing instances, got $(length(keys_list))"
    return keys_list
end

const ALL_KEYS = all_instance_keys()
const ALL_KEYS_SMOOTHING = all_instance_keys_smoothing()

# Smoothing regimes: classify from filename prefix boscia_<mu0>_<decay>_<mu_min>_
const SMOOTHING_REGIMES = ["large_mu", "small_mu", "decay_0.9", "decay_0.7"]
function smoothing_regime_from_filename(basename::String)::Union{String,Nothing}
    # Match boscia_<num>_<num>_<num>_E_optimality_...
    m = match(r"^boscia_([0-9.]+)_([0-9.]+)_([0-9.]+)_E_optimality", basename)
    m === nothing && return nothing
    decay = try; parse(Float64, m.captures[2]); catch; return nothing; end
    mu_min = try; parse(Float64, m.captures[3]); catch; return nothing; end
    if abs(decay - 1.0) < 1e-6
        return mu_min < 0.001 ? "large_mu" : "small_mu"
    end
    if abs(decay - 0.9) < 1e-6
        return "decay_0.9"
    end
    if abs(decay - 0.7) < 1e-6
        return "decay_0.7"
    end
    return nothing
end

function parse_smoothing_filename(basename::String)::Union{Tuple{Bool,Int,Int,Int,Int},Nothing}
    # ...E_optimality_(correlated|independent)__m_n_N_seed.csv
    m = match(r"E_optimality_(correlated|independent)__(\d+)_(\d+)_(\d+)_(\d+)\.csv$", basename)
    m === nothing && return nothing
    corr = m.captures[1] == "correlated"
    m_val = parse(Int, m.captures[2])
    n_val = parse(Int, m.captures[3])
    N_val = parse(Int, m.captures[4])
    seed_val = parse(Int, m.captures[5])
    return (corr, m_val, n_val, N_val, seed_val)
end

# ----- Boscia -----
const BOSCIA_DIR = joinpath(CSV_BASE, "Boscia")
const BOSCIA_PREFIX = "boscia__E_optimality_"
const BOSCIA_SUFFIX = "_cont__"
const BOSCIA_DELIM = ';'

function boscia_single_filename(corr::Bool, m::Int, n::Int, N::Int, seed::Int)
    type = corr ? "correlated" : "independent"
    return "boscia__E_optimality_$(type)_cont__$(m)_$(n)_$(N)_$(seed).csv"
end

function boscia_merged_filename(corr::Bool)
    type = corr ? "correlated" : "independent"
    return "boscia_E_optimality_$(type)_cont_merged.csv"
end

function read_boscia_single(path::String)::Union{DataFrame,Nothing}
    isfile(path) || return nothing
    try
        df = CSV.read(path, DataFrame; delim=BOSCIA_DELIM, silencewarnings=true)
        if nrow(df) >= 1
            return df[end:end, :]  # one row (last if multiple)
        end
        return nothing
    catch
        return nothing
    end
end

function boscia_placeholder_row(m::Int, n::Int, N::Int, seed::Int)
    return DataFrame(
        seed=seed,
        numberOfExperiments=m,
        numberOfParameters=n,
        N=N,
        time=TIME_LIMIT,
        solution=Inf,
        scaled_solution=Inf,
        dual_gap=0.0,
        rel_dual_gap=Inf,
        ncalls=0,
        num_nodes=0,
        termination="ERROR",
        optimal_time=0.0,
        optimal_iteration=0,
        solution_source="missing",
    )
end

function merge_boscia_group(corr::Bool; verbose=true)
    type_str = corr ? "correlated" : "independent"
    group_name = "Boscia E $type_str cont"
    if verbose
        println("\n--- $group_name ---")
    end
    key_to_row = Dict{Tuple{Int,Int,Int,Int}, DataFrame}()
    missing_list = Tuple{Int,Int,Int,Int}[]
    for (m, n, N, seed) in ALL_KEYS
        fname = boscia_single_filename(corr, m, n, N, seed)
        path = joinpath(BOSCIA_DIR, fname)
        df = read_boscia_single(path)
        if df !== nothing
            key_to_row[(m, n, N, seed)] = df
        else
            push!(missing_list, (m, n, N, seed))
            key_to_row[(m, n, N, seed)] = boscia_placeholder_row(m, n, N, seed)
        end
    end
    n_found = 75 - length(missing_list)
    if verbose
        println("Found $n_found/75 runs" * (isempty(missing_list) ? "" : " ($(length(missing_list)) missing)"))
    end
    if !isempty(missing_list) && verbose
        println("Missing instances:")
        for (m, n, N, seed) in sort!(missing_list)
            println("  m=$m n=$n N=$N seed=$seed")
        end
    end
    # Build merged DataFrame in fixed order
    rows = [key_to_row[k] for k in ALL_KEYS]
    merged = vcat(rows...)
    out_path = joinpath(BOSCIA_DIR, boscia_merged_filename(corr))
    CSV.write(out_path, merged; delim=BOSCIA_DELIM)
    if verbose
        println("Wrote $(nrow(merged)) rows -> $(out_path)")
    end
    return merged, missing_list
end

# ----- SCIP SDP -----
const SCIPSDP_DIR = joinpath(CSV_BASE, "SCIPSDP")
const SCIPSDP_PREFIX_OA = "scip_sdp_oa_E_optimality_"
const SCIPSDP_PREFIX_BNB = "scip_sdp_bnb_E_optimality_"
const SCIPSDP_SUFFIX = "_cont__"
const SCIPSDP_DELIM = ','

function scipsdp_single_filename(mode::String, corr::Bool, m::Int, n::Int, N::Int, seed::Int)
    type = corr ? "correlated" : "independent"
    prefix = mode == "oa" ? SCIPSDP_PREFIX_OA : SCIPSDP_PREFIX_BNB
    return "$(prefix)$(type)_cont__$(m)_$(n)_$(N)_$(seed).csv"
end

function scipsdp_merged_filename(mode::String, corr::Bool)
    type = corr ? "correlated" : "independent"
    return "scip_sdp_$(mode)_E_optimality_$(type)_cont_merged.csv"
end

function read_scipsdp_single(path::String)::Union{DataFrame,Nothing}
    isfile(path) || return nothing
    try
        df = CSV.read(path, DataFrame; delim=SCIPSDP_DELIM, silencewarnings=true)
        if nrow(df) >= 1
            return df[end:end, :]  # one row (last if multiple)
        end
        return nothing
    catch
        return nothing
    end
end

function scipsdp_placeholder_row(m::Int, n::Int, N::Int, seed::Int)
    return DataFrame(
        seed=seed,
        numberOfExperiments=m,
        numberOfParameters=n,
        time=TIME_LIMIT,
        N=N,
        solution=Inf,
        dual_bound=Inf,
        rel_gap=Inf,
        scaled_solution=Inf,
        termination="ERROR",
        feasible=false,
        n_nodes=0,
        n_cuts_found=0,
        n_cuts_applied=0,
        n_sdp_iters=0,
    )
end

function merge_scipsdp_group(mode::String, corr::Bool; verbose=true)
    type_str = corr ? "correlated" : "independent"
    group_name = "SCIPSDP $mode E $type_str cont"
    if verbose
        println("\n--- $group_name ---")
    end
    key_to_row = Dict{Tuple{Int,Int,Int,Int}, DataFrame}()
    missing_list = Tuple{Int,Int,Int,Int}[]
    for (m, n, N, seed) in ALL_KEYS
        fname = scipsdp_single_filename(mode, corr, m, n, N, seed)
        path = joinpath(SCIPSDP_DIR, fname)
        df = read_scipsdp_single(path)
        if df !== nothing
            key_to_row[(m, n, N, seed)] = df
        else
            push!(missing_list, (m, n, N, seed))
            key_to_row[(m, n, N, seed)] = scipsdp_placeholder_row(m, n, N, seed)
        end
    end
    n_found = 75 - length(missing_list)
    if verbose
        println("Found $n_found/75 runs" * (isempty(missing_list) ? "" : " ($(length(missing_list)) missing)"))
    end
    if !isempty(missing_list) && verbose
        println("Missing instances:")
        for (m, n, N, seed) in sort!(missing_list)
            println("  m=$m n=$n N=$N seed=$seed")
        end
    end
    rows = [key_to_row[k] for k in ALL_KEYS]
    merged = vcat(rows...)
    out_path = joinpath(SCIPSDP_DIR, scipsdp_merged_filename(mode, corr))
    CSV.write(out_path, merged; delim=SCIPSDP_DELIM)
    if verbose
        println("Wrote $(nrow(merged)) rows -> $(out_path)")
    end
    return merged, missing_list
end

# ----- Boscia smoothing (4 regimes) -----
function boscia_smoothing_merged_filename(regime::String, corr::Bool)
    type = corr ? "correlated" : "independent"
    return "boscia_smoothing_$(regime)_E_optimality_$(type)_merged.csv"
end

function merge_boscia_smoothing_group(regime::String, corr::Bool; verbose=true)
    type_str = corr ? "correlated" : "independent"
    group_name = "Boscia smoothing $regime $type_str"
    if verbose
        println("\n--- $group_name ---")
    end
    # Discover files in BOSCIA_DIR that belong to this (regime, corr)
    key_to_path = Dict{Tuple{Int,Int,Int,Int}, String}()
    for ent in readdir(BOSCIA_DIR; join=true)
        isfile(ent) || continue
        base = basename(ent)
        r = smoothing_regime_from_filename(base)
        r === nothing || r != regime && continue
        parsed = parse_smoothing_filename(base)
        parsed === nothing && continue
        (c, m, n, N, seed) = parsed
        c != corr && continue
        key = (m, n, N, seed)
        key_to_path[key] = ent
    end
    key_to_row = Dict{Tuple{Int,Int,Int,Int}, DataFrame}()
    missing_list = Tuple{Int,Int,Int,Int}[]
    for key in ALL_KEYS_SMOOTHING
        path = get(key_to_path, key, nothing)
        df = path !== nothing ? read_boscia_single(path) : nothing
        if df !== nothing
            key_to_row[key] = df
        else
            push!(missing_list, key)
            (m, n, N, seed) = key
            key_to_row[key] = boscia_placeholder_row(m, n, N, seed)
        end
    end
    n_found = 36 - length(missing_list)
    if verbose
        println("Found $n_found/36 runs" * (isempty(missing_list) ? "" : " ($(length(missing_list)) missing)"))
    end
    if !isempty(missing_list) && verbose
        println("Missing instances:")
        for key in sort!(missing_list)
            println("  m=$(key[1]) n=$(key[2]) N=$(key[3]) seed=$(key[4])")
        end
    end
    rows = [key_to_row[k] for k in ALL_KEYS_SMOOTHING]
    merged = vcat(rows...)
    out_path = joinpath(BOSCIA_DIR, boscia_smoothing_merged_filename(regime, corr))
    CSV.write(out_path, merged; delim=BOSCIA_DELIM)
    if verbose
        println("Wrote $(nrow(merged)) rows -> $(out_path)")
    end
    return merged, missing_list
end

# ----- Main -----
function run_merge(; solvers=nothing, verbose=true)
    if solvers === nothing
        solvers = ["Boscia", "SCIPSDP", "BosciaSmoothing"]
    end
    println("Merging single-run CSVs. Base: $CSV_BASE")
    println("Solvers: $(join(solvers, ", "))")
    for s in solvers
        if s == "Boscia"
            isdir(BOSCIA_DIR) || (println("Skip Boscia: $BOSCIA_DIR not found"); continue)
            println("(75 instances per group)")
            merge_boscia_group(true; verbose)
            merge_boscia_group(false; verbose)
        elseif s == "SCIPSDP"
            isdir(SCIPSDP_DIR) || (println("Skip SCIPSDP: $SCIPSDP_DIR not found"); continue)
            println("(75 instances per group)")
            merge_scipsdp_group("oa", true; verbose)
            merge_scipsdp_group("oa", false; verbose)
            merge_scipsdp_group("bnb", true; verbose)
            merge_scipsdp_group("bnb", false; verbose)
        elseif s == "BosciaSmoothing"
            isdir(BOSCIA_DIR) || (println("Skip BosciaSmoothing: $BOSCIA_DIR not found"); continue)
            println("(36 instances per group, 4 regimes)")
            for regime in SMOOTHING_REGIMES
                merge_boscia_smoothing_group(regime, true; verbose)
                merge_boscia_smoothing_group(regime, false; verbose)
            end
        else
            println("Unknown solver: $s")
        end
    end
    println("\nDone. Single-run CSV files were not modified.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    solvers = isempty(ARGS) ? nothing : ARGS
    run_merge(; solvers)
end
