#!/usr/bin/env julia
#=
Merge single-run CSVs into one CSV per group (solver + mode + data type + correlated/independent).
- Keeps all single-run CSV files unchanged.
- E-optimal / SCIP: 75 runs per group (dimensions 50, 80, 100, 120, 150 × 5 seeds × 3 N values).
- Boscia smoothing: 36 runs per group (dimensions 50, 100, 150, 200 × 3 seeds × 3 N values);
  four regimes: large_mu (decay=1, μ_min<0.001), small_mu (decay=1, μ_min≥0.001), decay_0.9, decay_0.7.
- AGC (algebraic graph connectivity): 20 runs per group (4 dimensions × 5 seeds), setups
  correlated_connected and independent_disconnected, inferred from filenames. Also merges
  rank_based_pruning and exclusion-criterion variants when single-run CSVs exist.
- Missing instances get a placeholder row: time=3600, solution/scaled_solution=Inf, termination=ERROR,
  statistics = 0. Prints which instances are missing per group.

- ACST / ACSTS (algebraic connectivity spanning tree, Boscia, independent): 21 runs per merged file
  (7 edge-counts × 3 seeds; N = n−1). Default merge target ACSTTrees: ACST = baseline + rank pruning;
  ACSTS = baseline + rank pruning + exclusion_criterion; **Pajarito** merged CSV for **ACST** only
  (`pajarito_ACST_optimality_independent_merged.csv`) for comparison with Boscia (same instance grid).
  BosciaExclusion also merges all exclusion folders for both criteria.

Usage: julia merge_single_runs_to_csv.jl [Boscia] [BosciaExclusion] [SCIPSDP] [BosciaSmoothing] [AGC] [ACSTTrees]
  No args = process all listed targets including ACSTTrees.
=#

using CSV, DataFrames

const CSV_BASE = joinpath(@__DIR__, "csv")
const PAJARITO_DIR = joinpath(CSV_BASE, "Pajarito")
const PAJARITO_DELIM = ','
const TIME_LIMIT = 3600
const SEEDS = 1:5
const DIMENSIONS_M = [50, 80, 100, 120, 150]
# Smoothing experiment: 4 dims × 3 seeds × 3 N = 36
const DIMENSIONS_SMOOTHING = [50, 100, 150, 200]
const SEEDS_SMOOTHING = 1:3
# AGC: dimensions 80,100,150,200 (50 is ignored / unusable)
const DIMENSIONS_AGC = [80, 100, 150, 200]
# ACST / ACSTS: m = |E|(K_n) = n(n-1)/2 for n in {10,12,15,25,40,60,100}
const DIMENSIONS_ACST_M = [45, 66, 105, 300, 780, 1770, 4950]
const SEEDS_ACST = 1:3

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

# ----- Boscia E-optimality (default table now uses reduced_spectrum data under the "Boscia" name) -----
const BOSCIA_DIR = joinpath(CSV_BASE, "Boscia")
# Single-run: boscia__E_optimality_<type>__m_n_N_seed.csv (connection empty for E-opt)
const BOSCIA_DELIM = ';'
const BOSCIA_DEFAULT_E_FOLDER = "reduced_spectrum"
const BOSCIA_DEFAULT_E_SUBDIR = "reduced_spectrum_half"
const BOSCIA_DEFAULT_AGC_FOLDER = "baseline"
const BOSCIA_DEFAULT_ACST_FOLDER = "rank_based_pruning"

function boscia_single_filename(corr::Bool, m::Int, n::Int, N::Int, seed::Int)
    type = corr ? "correlated" : "independent"
    return "boscia__E_optimality_$(type)__$(m)_$(n)_$(N)_$(seed).csv"
end

function boscia_prefixed_single_filename(folder::String, corr::Bool, m::Int, n::Int, N::Int, seed::Int)
    type = corr ? "correlated" : "independent"
    return "boscia_$(folder)_E_optimality_$(type)__$(m)_$(n)_$(N)_$(seed).csv"
end

function boscia_merged_filename(corr::Bool)
    type = corr ? "correlated" : "independent"
    return "boscia_E_optimality_$(type)_cont_merged.csv"
end

function boscia_baseline_merged_filename(corr::Bool)
    type = corr ? "correlated" : "independent"
    return "boscia_baseline_E_optimality_$(type)_merged.csv"
end

# ----- Boscia exclusion criterion (E-optimality, same 75 instances) -----
const BOSCIA_EXCLUSION_FOLDERS = [
    "exclusion_criterion",
    "exclusion_criterion_random",
    "exclusion_criterion_tighter_tol",
    "dual_exclusion_criterion",
]

function boscia_exclusion_single_filename(exclusion_folder::String, corr::Bool, m::Int, n::Int, N::Int, seed::Int)
    type = corr ? "correlated" : "independent"
    return "boscia_$(exclusion_folder)_E_optimality_$(type)__$(m)_$(n)_$(N)_$(seed).csv"
end

function boscia_exclusion_merged_filename(exclusion_folder::String, corr::Bool)
    type = corr ? "correlated" : "independent"
    return "boscia_$(exclusion_folder)_E_optimality_$(type)_merged.csv"
end

function merge_boscia_exclusion_group(exclusion_folder::String, corr::Bool; verbose=true)
    type_str = corr ? "correlated" : "independent"
    group_name = "Boscia E ($exclusion_folder) $type_str"
    if verbose
        println("\n--- $group_name ---")
    end
    key_to_row = Dict{Tuple{Int,Int,Int,Int}, DataFrame}()
    missing_list = Tuple{Int,Int,Int,Int}[]
    found_any = false
    for (m, n, N, seed) in ALL_KEYS
        fname = boscia_exclusion_single_filename(exclusion_folder, corr, m, n, N, seed)
        path = joinpath(BOSCIA_DIR, fname)
        df = read_boscia_single(path)
        if df !== nothing
            found_any = true
            key_to_row[(m, n, N, seed)] = df
        else
            push!(missing_list, (m, n, N, seed))
            key_to_row[(m, n, N, seed)] = boscia_placeholder_row(m, n, N, seed)
        end
    end

    if !found_any
        if verbose
            println("No single-run CSVs found for $group_name; skipping merged output.")
        end
        return nothing, missing_list
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
    # Different exclusion variants may yield different column sets; merge by column union.
    merged = vcat(rows...; cols=:union)
    out_path = joinpath(BOSCIA_DIR, boscia_exclusion_merged_filename(exclusion_folder, corr))
    CSV.write(out_path, merged; delim=BOSCIA_DELIM)
    if verbose
        println("Wrote $(nrow(merged)) rows -> $(out_path)")
    end
    return merged, missing_list
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
        # Some Boscia runs include these tightening statistics (dual exclusion variants).
        # Placeholders must carry the same columns so vcat across runs does not error.
        avg_fixed_to_one=0.0,
        avg_fixed_to_zero=0.0,
        solution_source="missing",
    )
end

function merge_boscia_group(corr::Bool; verbose=true)
    type_str = corr ? "correlated" : "independent"
    source_label = corr ? "baseline" : BOSCIA_DEFAULT_E_FOLDER
    group_name = "Boscia E $type_str cont (from $(source_label))"
    if verbose
        println("\n--- $group_name ---")
    end
    key_to_row = Dict{Tuple{Int,Int,Int,Int}, DataFrame}()
    missing_list = Tuple{Int,Int,Int,Int}[]
    for (m, n, N, seed) in ALL_KEYS
        path = if corr
            joinpath(BOSCIA_DIR, boscia_single_filename(corr, m, n, N, seed))
        else
            fname = boscia_prefixed_single_filename(BOSCIA_DEFAULT_E_FOLDER, corr, m, n, N, seed)
            p = joinpath(BOSCIA_DIR, BOSCIA_DEFAULT_E_SUBDIR, fname)
            if !isfile(p)
                # Fallback for legacy layout where reduced_spectrum files are directly in BOSCIA_DIR.
                p = joinpath(BOSCIA_DIR, fname)
            end
            p
        end
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
    merged = vcat(rows...; cols=:union)
    out_path = joinpath(BOSCIA_DIR, boscia_merged_filename(corr))
    CSV.write(out_path, merged; delim=BOSCIA_DELIM)
    if verbose
        println("Wrote $(nrow(merged)) rows -> $(out_path)")
    end
    return merged, missing_list
end

function merge_boscia_baseline_group(corr::Bool; verbose=true)
    type_str = corr ? "correlated" : "independent"
    group_name = "Boscia baseline E $type_str"
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
    rows = [key_to_row[k] for k in ALL_KEYS]
    merged = vcat(rows...; cols=:union)
    out_path = joinpath(BOSCIA_DIR, boscia_baseline_merged_filename(corr))
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
    return "$(prefix)$(type)__$(m)_$(n)_$(N)_$(seed).csv"
end

function scipsdp_merged_filename(mode::String, corr::Bool)
    type = corr ? "correlated" : "independent"
    return "scip_sdp_$(mode)_E_optimality_$(type)_merged.csv"
end

function scipsdp_acst_single_filename(mode::String, m::Int, n::Int, N::Int, seed::Int)
    return "scip_sdp_$(mode)_ACST_optimality_independent__$(m)_$(n)_$(N)_$(seed).csv"
end

function scipsdp_acst_merged_filename(mode::String)
    return "scip_sdp_$(mode)_ACST_optimality_independent_merged.csv"
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
    merged = vcat(rows...; cols=:union)
    out_path = joinpath(SCIPSDP_DIR, scipsdp_merged_filename(mode, corr))
    CSV.write(out_path, merged; delim=SCIPSDP_DELIM)
    if verbose
        println("Wrote $(nrow(merged)) rows -> $(out_path)")
    end
    return merged, missing_list
end

function merge_scipsdp_acst_group(mode::String; verbose=true)
    if verbose
        println("\n--- SCIPSDP $mode ACST ($(length(ALL_KEYS_ACST)) instances) ---")
    end
    key_to_row = Dict{Tuple{Int,Int,Int,Int}, DataFrame}()
    missing_list = Tuple{Int,Int,Int,Int}[]
    found_any = false
    for (m, n, N, seed) in ALL_KEYS_ACST
        fname = scipsdp_acst_single_filename(mode, m, n, N, seed)
        path = joinpath(SCIPSDP_DIR, fname)
        df = read_scipsdp_single(path)
        if df !== nothing
            found_any = true
            key_to_row[(m, n, N, seed)] = df
        else
            push!(missing_list, (m, n, N, seed))
            key_to_row[(m, n, N, seed)] = scipsdp_placeholder_row(m, n, N, seed)
        end
    end
    if !found_any
        verbose && println("No single-run SCIPSDP $mode ACST CSVs found; skipping merged output.")
        return nothing, missing_list
    end
    n_expected = length(ALL_KEYS_ACST)
    n_found = n_expected - length(missing_list)
    if verbose
        println("Found $n_found/$n_expected runs" * (isempty(missing_list) ? "" : " ($(length(missing_list)) missing)"))
    end
    if !isempty(missing_list) && verbose
        println("Missing instances:")
        for (m, n, N, seed) in sort!(missing_list)
            println("  m=$m n=$n N=$N seed=$seed")
        end
    end
    rows = [key_to_row[k] for k in ALL_KEYS_ACST]
    merged = vcat(rows...; cols=:union)
    out_path = joinpath(SCIPSDP_DIR, scipsdp_acst_merged_filename(mode))
    CSV.write(out_path, merged; delim=SCIPSDP_DELIM)
    verbose && println("Wrote $(nrow(merged)) rows -> $(out_path)")
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
    merged = vcat(rows...; cols=:union)
    out_path = joinpath(BOSCIA_DIR, boscia_smoothing_merged_filename(regime, corr))
    CSV.write(out_path, merged; delim=BOSCIA_DELIM)
    if verbose
        println("Wrote $(nrow(merged)) rows -> $(out_path)")
    end
    return merged, missing_list
end

# ----- AGC (algebraic graph connectivity, Boscia + SCIPSDP) -----

function infer_agc_nN()
    mapping = Dict{Int,Tuple{Int,Int}}()  # m -> (n, N)
    # Boscia filenames: boscia__AGC_optimality_... or boscia_exclusion_criterion_AGC_optimality_...
    if isdir(BOSCIA_DIR)
        for ent in readdir(BOSCIA_DIR; join=true)
            isfile(ent) || continue
            base = basename(ent)
            m = match(r"^boscia__AGC_optimality_(correlated|independent)_(connected|disconnected)_(\d+)_(\d+)_(\d+)_(\d+)\.csv$", base)
            if m === nothing
                # e.g. boscia_<folder>_AGC_optimality_<type>_<connected|disconnected>_<m>_<n>_<N>_<seed>.csv
                m = match(r"^boscia_([A-Za-z0-9_]+)_AGC_optimality_(correlated|independent)_(connected|disconnected)_(\d+)_(\d+)_(\d+)_(\d+)\.csv$", base)
            end
            m === nothing && continue
            # Standard boscia__... regex: captures[1]=type, [2]=conn, [3]=m, [4]=n, [5]=N, [6]=seed
            # Generic boscia_<folder>... regex: captures[1]=folder, [2]=type, [3]=conn, [4]=m, [5]=n, [6]=N, [7]=seed
            m_val = parse(Int, length(m.captures) >= 7 ? m.captures[4] : m.captures[3])
            n_val = parse(Int, length(m.captures) >= 7 ? m.captures[5] : m.captures[4])
            N_val = parse(Int, length(m.captures) >= 7 ? m.captures[6] : m.captures[5])
            haskey(mapping, m_val) || (mapping[m_val] = (n_val, N_val))
        end
    end
    # SCIPSDP filenames: scip_sdp_{oa|bnb}_AGC_optimality_type_conn_m_n_N_seed.csv
    if isdir(SCIPSDP_DIR)
        for ent in readdir(SCIPSDP_DIR; join=true)
            isfile(ent) || continue
            base = basename(ent)
            m = match(r"^scip_sdp_(oa|bnb)_AGC_optimality_(correlated|independent)_(connected|disconnected)_(\d+)_(\d+)_(\d+)_(\d+)\.csv$", base)
            m === nothing && continue
            m_val = parse(Int, m.captures[4])
            n_val = parse(Int, m.captures[5])
            N_val = parse(Int, m.captures[6])
            haskey(mapping, m_val) || (mapping[m_val] = (n_val, N_val))
        end
    end
    for m in DIMENSIONS_AGC
        haskey(mapping, m) || error("AGC: could not infer (n, N) for m = $m from existing CSVs")
    end
    return mapping
end

function all_instance_keys_agc()
    dims_to_nN = infer_agc_nN()
    keys_list = Tuple{Int,Int,Int,Int}[]  # (m, n, N, seed)
    for m in DIMENSIONS_AGC
        (n, N) = dims_to_nN[m]
        for seed in SEEDS
            push!(keys_list, (m, n, N, seed))
        end
    end
    @assert length(keys_list) == length(DIMENSIONS_AGC) * length(SEEDS) "expected $(length(DIMENSIONS_AGC) * length(SEEDS)) AGC instances, got $(length(keys_list))"
    return keys_list
end

const ALL_KEYS_AGC = all_instance_keys_agc()

function boscia_agc_single_filename(corr::Bool, connected::Bool, m::Int, n::Int, N::Int, seed::Int)
    type = corr ? "correlated" : "independent"
    conn = connected ? "connected" : "disconnected"
    return "boscia__AGC_optimality_$(type)_$(conn)_$(m)_$(n)_$(N)_$(seed).csv"
end

function boscia_agc_prefixed_single_filename(folder::String, corr::Bool, connected::Bool, m::Int, n::Int, N::Int, seed::Int)
    type = corr ? "correlated" : "independent"
    conn = connected ? "connected" : "disconnected"
    return "boscia_$(folder)_AGC_optimality_$(type)_$(conn)_$(m)_$(n)_$(N)_$(seed).csv"
end

function boscia_agc_merged_filename(corr::Bool, connected::Bool)
    type = corr ? "correlated" : "independent"
    conn = connected ? "connected" : "disconnected"
    return "boscia_AGC_optimality_$(type)_$(conn)_merged.csv"
end

function boscia_agc_baseline_merged_filename(corr::Bool, connected::Bool)
    type = corr ? "correlated" : "independent"
    conn = connected ? "connected" : "disconnected"
    return "boscia_baseline_AGC_optimality_$(type)_$(conn)_merged.csv"
end

# Boscia AGC with exclusion criterion (same instance set as standard AGC)
function boscia_agc_exclusion_single_filename(exclusion_folder::String, corr::Bool, connected::Bool, m::Int, n::Int, N::Int, seed::Int)
    type = corr ? "correlated" : "independent"
    conn = connected ? "connected" : "disconnected"
    return "boscia_$(exclusion_folder)_AGC_optimality_$(type)_$(conn)_$(m)_$(n)_$(N)_$(seed).csv"
end

function boscia_agc_exclusion_merged_filename(exclusion_folder::String, corr::Bool, connected::Bool)
    type = corr ? "correlated" : "independent"
    conn = connected ? "connected" : "disconnected"
    return "boscia_$(exclusion_folder)_AGC_optimality_$(type)_$(conn)_merged.csv"
end

function merge_boscia_agc_exclusion_group(exclusion_folder::String, corr::Bool, connected::Bool; verbose=true)
    type_str = corr ? "correlated" : "independent"
    conn_str = connected ? "connected" : "disconnected"
    group_name = "Boscia AGC ($exclusion_folder) $type_str $conn_str"
    if verbose
        println("\n--- $group_name ---")
    end
    key_to_row = Dict{Tuple{Int,Int,Int,Int}, DataFrame}()
    missing_list = Tuple{Int,Int,Int,Int}[]
    found_any = false
    for (m, n, N, seed) in ALL_KEYS_AGC
        fname = boscia_agc_exclusion_single_filename(exclusion_folder, corr, connected, m, n, N, seed)
        path = joinpath(BOSCIA_DIR, fname)
        df = read_boscia_single(path)
        if df !== nothing
            found_any = true
            key_to_row[(m, n, N, seed)] = df
        else
            push!(missing_list, (m, n, N, seed))
            key_to_row[(m, n, N, seed)] = boscia_placeholder_row(m, n, N, seed)
        end
    end
    n_expected = length(ALL_KEYS_AGC)

    if !found_any
        if verbose
            println("No single-run CSVs found for $group_name; skipping merged output.")
        end
        return nothing, missing_list
    end

    n_found = n_expected - length(missing_list)
    if verbose
        println("Found $n_found/$n_expected runs" * (isempty(missing_list) ? "" : " ($(length(missing_list)) missing)"))
    end
    if !isempty(missing_list) && verbose
        println("Missing instances:")
        for (m, n, N, seed) in sort!(missing_list)
            println("  m=$m n=$n N=$N seed=$seed")
        end
    end
    rows = [key_to_row[k] for k in ALL_KEYS_AGC]
    merged = vcat(rows...; cols=:union)
    out_path = joinpath(BOSCIA_DIR, boscia_agc_exclusion_merged_filename(exclusion_folder, corr, connected))
    CSV.write(out_path, merged; delim=BOSCIA_DELIM)
    if verbose
        println("Wrote $(nrow(merged)) rows -> $(out_path)")
    end
    return merged, missing_list
end

function merge_boscia_agc_group(connected::Bool, corr::Bool; verbose=true)
    type_str = corr ? "correlated" : "independent"
    conn_str = connected ? "connected" : "disconnected"
    group_name = "Boscia AGC $type_str $conn_str (baseline)"
    if verbose
        println("\n--- $group_name ---")
    end
    key_to_row = Dict{Tuple{Int,Int,Int,Int}, DataFrame}()
    missing_list = Tuple{Int,Int,Int,Int}[]
    for (m, n, N, seed) in ALL_KEYS_AGC
        fname = boscia_agc_single_filename(corr, connected, m, n, N, seed)
        path = joinpath(BOSCIA_DIR, fname)
        df = read_boscia_single(path)
        if df !== nothing
            key_to_row[(m, n, N, seed)] = df
        else
            push!(missing_list, (m, n, N, seed))
            key_to_row[(m, n, N, seed)] = boscia_placeholder_row(m, n, N, seed)
        end
    end
    n_expected = length(ALL_KEYS_AGC)
    n_found = n_expected - length(missing_list)
    if verbose
        println("Found $n_found/$n_expected runs" * (isempty(missing_list) ? "" : " ($(length(missing_list)) missing)"))
    end
    if !isempty(missing_list) && verbose
        println("Missing instances:")
        for (m, n, N, seed) in sort!(missing_list)
            println("  m=$m n=$n N=$N seed=$seed")
        end
    end
    rows = [key_to_row[k] for k in ALL_KEYS_AGC]
    merged = vcat(rows...; cols=:union)
    out_path = joinpath(BOSCIA_DIR, boscia_agc_merged_filename(corr, connected))
    CSV.write(out_path, merged; delim=BOSCIA_DELIM)
    if verbose
        println("Wrote $(nrow(merged)) rows -> $(out_path)")
    end
    return merged, missing_list
end

function merge_boscia_agc_baseline_group(connected::Bool, corr::Bool; verbose=true)
    type_str = corr ? "correlated" : "independent"
    conn_str = connected ? "connected" : "disconnected"
    group_name = "Boscia baseline AGC $type_str $conn_str"
    if verbose
        println("\n--- $group_name ---")
    end
    key_to_row = Dict{Tuple{Int,Int,Int,Int}, DataFrame}()
    missing_list = Tuple{Int,Int,Int,Int}[]
    for (m, n, N, seed) in ALL_KEYS_AGC
        fname = boscia_agc_single_filename(corr, connected, m, n, N, seed)
        path = joinpath(BOSCIA_DIR, fname)
        df = read_boscia_single(path)
        if df !== nothing
            key_to_row[(m, n, N, seed)] = df
        else
            push!(missing_list, (m, n, N, seed))
            key_to_row[(m, n, N, seed)] = boscia_placeholder_row(m, n, N, seed)
        end
    end
    n_expected = length(ALL_KEYS_AGC)
    n_found = n_expected - length(missing_list)
    if verbose
        println("Found $n_found/$n_expected runs" * (isempty(missing_list) ? "" : " ($(length(missing_list)) missing)"))
    end
    if !isempty(missing_list) && verbose
        println("Missing instances:")
        for (m, n, N, seed) in sort!(missing_list)
            println("  m=$m n=$n N=$N seed=$seed")
        end
    end
    rows = [key_to_row[k] for k in ALL_KEYS_AGC]
    merged = vcat(rows...; cols=:union)
    out_path = joinpath(BOSCIA_DIR, boscia_agc_baseline_merged_filename(corr, connected))
    CSV.write(out_path, merged; delim=BOSCIA_DELIM)
    if verbose
        println("Wrote $(nrow(merged)) rows -> $(out_path)")
    end
    return merged, missing_list
end

function scipsdp_agc_single_filename(mode::String, corr::Bool, connected::Bool, m::Int, n::Int, N::Int, seed::Int)
    type = corr ? "correlated" : "independent"
    conn = connected ? "connected" : "disconnected"
    prefix = mode == "oa" ? "scip_sdp_oa" : "scip_sdp_bnb"
    return "$(prefix)_AGC_optimality_$(type)_$(conn)_$(m)_$(n)_$(N)_$(seed).csv"
end

function scipsdp_agc_merged_filename(mode::String, corr::Bool, connected::Bool)
    type = corr ? "correlated" : "independent"
    conn = connected ? "connected" : "disconnected"
    return "scip_sdp_$(mode)_AGC_optimality_$(type)_$(conn)_merged.csv"
end

function merge_scipsdp_agc_group(mode::String, connected::Bool, corr::Bool; verbose=true)
    type_str = corr ? "correlated" : "independent"
    conn_str = connected ? "connected" : "disconnected"
    group_name = "SCIPSDP $mode AGC $type_str $conn_str"
    if verbose
        println("\n--- $group_name ---")
    end
    key_to_row = Dict{Tuple{Int,Int,Int,Int}, DataFrame}()
    missing_list = Tuple{Int,Int,Int,Int}[]
    for (m, n, N, seed) in ALL_KEYS_AGC
        fname = scipsdp_agc_single_filename(mode, corr, connected, m, n, N, seed)
        path = joinpath(SCIPSDP_DIR, fname)
        df = read_scipsdp_single(path)
        if df !== nothing
            key_to_row[(m, n, N, seed)] = df
        else
            push!(missing_list, (m, n, N, seed))
            key_to_row[(m, n, N, seed)] = scipsdp_placeholder_row(m, n, N, seed)
        end
    end
    n_expected = length(ALL_KEYS_AGC)
    n_found = n_expected - length(missing_list)
    if verbose
        println("Found $n_found/$n_expected runs" * (isempty(missing_list) ? "" : " ($(length(missing_list)) missing)"))
    end
    if !isempty(missing_list) && verbose
        println("Missing instances:")
        for (m, n, N, seed) in sort!(missing_list)
            println("  m=$m n=$n N=$N seed=$seed")
        end
    end
    rows = [key_to_row[k] for k in ALL_KEYS_AGC]
    merged = vcat(rows...; cols=:union)
    out_path = joinpath(SCIPSDP_DIR, scipsdp_agc_merged_filename(mode, corr, connected))
    CSV.write(out_path, merged; delim=SCIPSDP_DELIM)
    if verbose
        println("Wrote $(nrow(merged)) rows -> $(out_path)")
    end
    return merged, missing_list
end

# ----- ACST / ACSTS (Boscia, independent, spanning-tree instances) -----
function acst_n_from_edges(m_edges::Int)
    disc = 1 + 8 * m_edges
    sn = round(Int, sqrt(disc))
    n = (1 + sn) ÷ 2
    n * (n - 1) ÷ 2 == m_edges || error("ACST: m=$m_edges is not n(n-1)/2 for integer n")
    return n
end

function all_instance_keys_acst()
    keys = Tuple{Int,Int,Int,Int}[]
    for m in DIMENSIONS_ACST_M
        n = acst_n_from_edges(m)
        N = n - 1
        for seed in SEEDS_ACST
            push!(keys, (m, n, N, seed))
        end
    end
    return keys
end

const ALL_KEYS_ACST = all_instance_keys_acst()

function boscia_acst_baseline_single_filename(criterion::String, m::Int, n::Int, N::Int, seed::Int)
    return "boscia__$(criterion)_optimality_independent__$(m)_$(n)_$(N)_$(seed).csv"
end

function boscia_acst_prefixed_single_filename(folder::String, criterion::String, m::Int, n::Int, N::Int, seed::Int)
    return "boscia_$(folder)_$(criterion)_optimality_independent__$(m)_$(n)_$(N)_$(seed).csv"
end

function merge_boscia_acst_variant(criterion::String, folder::Union{Nothing,String}; verbose=true)
    label = folder === nothing ? "baseline" : folder
    if verbose
        println("\n--- Boscia $criterion ($label, $(length(ALL_KEYS_ACST)) instances) ---")
    end
    key_to_row = Dict{Tuple{Int,Int,Int,Int}, DataFrame}()
    missing_list = Tuple{Int,Int,Int,Int}[]
    found_any = false
    for (m, n, N, seed) in ALL_KEYS_ACST
        fname = if folder === nothing
            boscia_acst_baseline_single_filename(criterion, m, n, N, seed)
        else
            boscia_acst_prefixed_single_filename(folder, criterion, m, n, N, seed)
        end
        path = joinpath(BOSCIA_DIR, fname)
        df = read_boscia_single(path)
        if df !== nothing
            found_any = true
            key_to_row[(m, n, N, seed)] = df
        else
            push!(missing_list, (m, n, N, seed))
            key_to_row[(m, n, N, seed)] = boscia_placeholder_row(m, n, N, seed)
        end
    end
    if !found_any
        if verbose
            println("No single-run CSVs found; skipping merged output.")
        end
        return nothing, missing_list
    end
    n_expected = length(ALL_KEYS_ACST)
    n_found = n_expected - length(missing_list)
    if verbose
        println("Found $n_found/$n_expected runs" * (isempty(missing_list) ? "" : " ($(length(missing_list)) missing)"))
    end
    if !isempty(missing_list) && verbose
        println("Missing instances:")
        for (m, n, N, seed) in sort!(missing_list)
            println("  m=$m n=$n N=$N seed=$seed")
        end
    end
    rows = [key_to_row[k] for k in ALL_KEYS_ACST]
    merged = vcat(rows...; cols=:union)
    out_path = if folder === nothing
        joinpath(BOSCIA_DIR, "boscia_$(criterion)_optimality_independent_merged.csv")
    else
        joinpath(BOSCIA_DIR, "boscia_$(folder)_$(criterion)_optimality_independent_merged.csv")
    end
    CSV.write(out_path, merged; delim=BOSCIA_DELIM)
    if verbose
        println("Wrote $(nrow(merged)) rows -> $(out_path)")
    end
    return merged, missing_list
end

function merge_boscia_acst_default_variant(criterion::String; verbose=true)
    label = BOSCIA_DEFAULT_ACST_FOLDER
    if verbose
        println("\n--- Boscia $criterion ($label, $(length(ALL_KEYS_ACST)) instances) ---")
    end
    key_to_row = Dict{Tuple{Int,Int,Int,Int}, DataFrame}()
    missing_list = Tuple{Int,Int,Int,Int}[]
    found_any = false
    for (m, n, N, seed) in ALL_KEYS_ACST
        fname = boscia_acst_prefixed_single_filename(BOSCIA_DEFAULT_ACST_FOLDER, criterion, m, n, N, seed)
        path = joinpath(BOSCIA_DIR, fname)
        df = read_boscia_single(path)
        if df !== nothing
            found_any = true
            key_to_row[(m, n, N, seed)] = df
        else
            push!(missing_list, (m, n, N, seed))
            key_to_row[(m, n, N, seed)] = boscia_placeholder_row(m, n, N, seed)
        end
    end
    if !found_any
        verbose && println("No single-run CSVs found; skipping merged output.")
        return nothing, missing_list
    end
    n_expected = length(ALL_KEYS_ACST)
    n_found = n_expected - length(missing_list)
    if verbose
        println("Found $n_found/$n_expected runs" * (isempty(missing_list) ? "" : " ($(length(missing_list)) missing)"))
    end
    rows = [key_to_row[k] for k in ALL_KEYS_ACST]
    merged = vcat(rows...; cols=:union)
    out_path = joinpath(BOSCIA_DIR, "boscia_$(criterion)_optimality_independent_merged.csv")
    CSV.write(out_path, merged; delim=BOSCIA_DELIM)
    verbose && println("Wrote $(nrow(merged)) rows -> $(out_path)")
    return merged, missing_list
end

function merge_boscia_acst_baseline_variant(criterion::String; verbose=true)
    label = "baseline"
    if verbose
        println("\n--- Boscia $criterion ($label, $(length(ALL_KEYS_ACST)) instances) ---")
    end
    key_to_row = Dict{Tuple{Int,Int,Int,Int}, DataFrame}()
    missing_list = Tuple{Int,Int,Int,Int}[]
    found_any = false
    for (m, n, N, seed) in ALL_KEYS_ACST
        fname = boscia_acst_baseline_single_filename(criterion, m, n, N, seed)
        path = joinpath(BOSCIA_DIR, fname)
        df = read_boscia_single(path)
        if df !== nothing
            found_any = true
            key_to_row[(m, n, N, seed)] = df
        else
            push!(missing_list, (m, n, N, seed))
            key_to_row[(m, n, N, seed)] = boscia_placeholder_row(m, n, N, seed)
        end
    end
    if !found_any
        verbose && println("No single-run CSVs found; skipping merged output.")
        return nothing, missing_list
    end
    n_expected = length(ALL_KEYS_ACST)
    n_found = n_expected - length(missing_list)
    if verbose
        println("Found $n_found/$n_expected runs" * (isempty(missing_list) ? "" : " ($(length(missing_list)) missing)"))
    end
    rows = [key_to_row[k] for k in ALL_KEYS_ACST]
    merged = vcat(rows...; cols=:union)
    out_path = joinpath(BOSCIA_DIR, "boscia_baseline_$(criterion)_optimality_independent_merged.csv")
    CSV.write(out_path, merged; delim=BOSCIA_DELIM)
    verbose && println("Wrote $(nrow(merged)) rows -> $(out_path)")
    return merged, missing_list
end

function pajarito_acst_single_filename(m::Int, n::Int, N::Int, seed::Int)
    return "pajarito_ACST_optimality_independent_$(m)_$(n)_$(N)_$(seed).csv"
end

function read_pajarito_acst_single(path::String)::Union{DataFrame,Nothing}
    isfile(path) || return nothing
    try
        df = CSV.read(path, DataFrame; delim=PAJARITO_DELIM, silencewarnings=true)
        return nrow(df) >= 1 ? df[end:end, :] : nothing
    catch
        return nothing
    end
end

function pajarito_acst_placeholder_row(m::Int, n::Int, N::Int, seed::Int)
    return DataFrame(
        seed=seed,
        numberOfExperiments=m,
        numberOfParameters=n,
        time=TIME_LIMIT,
        N=N,
        solution=Inf,
        scaled_solution=Inf,
        termination="ERROR",
        numberIterations=0,
        numberCuts=0,
        feasible=false,
    )
end

function merge_pajarito_acst_group(; verbose=true)
    if verbose
        println("\n--- Pajarito ACST ($(length(ALL_KEYS_ACST)) instances) ---")
    end
    key_to_row = Dict{Tuple{Int,Int,Int,Int}, DataFrame}()
    missing_list = Tuple{Int,Int,Int,Int}[]
    found_any = false
    for (m, n, N, seed) in ALL_KEYS_ACST
        fname = pajarito_acst_single_filename(m, n, N, seed)
        path = joinpath(PAJARITO_DIR, fname)
        df = read_pajarito_acst_single(path)
        if df !== nothing
            found_any = true
            key_to_row[(m, n, N, seed)] = df
        else
            push!(missing_list, (m, n, N, seed))
            key_to_row[(m, n, N, seed)] = pajarito_acst_placeholder_row(m, n, N, seed)
        end
    end
    if !found_any
        verbose && println("No single-run Pajarito ACST CSVs found; skipping merged output.")
        return nothing, missing_list
    end
    n_expected = length(ALL_KEYS_ACST)
    n_found = n_expected - length(missing_list)
    if verbose
        println("Found $n_found/$n_expected runs" * (isempty(missing_list) ? "" : " ($(length(missing_list)) missing)"))
    end
    if !isempty(missing_list) && verbose
        println("Missing instances:")
        for (m, n, N, seed) in sort!(missing_list)
            println("  m=$m n=$n N=$N seed=$seed")
        end
    end
    rows = [key_to_row[k] for k in ALL_KEYS_ACST]
    merged = vcat(rows...; cols=:union)
    out_path = joinpath(PAJARITO_DIR, "pajarito_ACST_optimality_independent_merged.csv")
    CSV.write(out_path, merged; delim=PAJARITO_DELIM)
    verbose && println("Wrote $(nrow(merged)) rows -> $(out_path)")
    return merged, missing_list
end

# ----- Main -----
function run_merge(; solvers=nothing, verbose=true)
    if solvers === nothing
        solvers = ["Boscia", "BosciaExclusion", "SCIPSDP", "BosciaSmoothing", "AGC", "ACSTTrees"]
    end
    println("Merging single-run CSVs. Base: $CSV_BASE")
    println("Solvers: $(join(solvers, ", "))")
    for s in solvers
        if s == "Boscia"
            isdir(BOSCIA_DIR) || (println("Skip Boscia: $BOSCIA_DIR not found"); continue)
            println("(75 instances per group)")
            merge_boscia_group(true; verbose)
            merge_boscia_group(false; verbose)
            println("(baseline reference, 75 instances per group)")
            merge_boscia_baseline_group(true; verbose)
            merge_boscia_baseline_group(false; verbose)
            # Rank-based pruning uses the same instance grid as exclusion variants (opt_design_boscia folder prefix).
            println("(rank-based pruning, 75 instances per group)")
            merge_boscia_exclusion_group("rank_based_pruning", true; verbose)
            merge_boscia_exclusion_group("rank_based_pruning", false; verbose)
        elseif s == "BosciaExclusion"
            isdir(BOSCIA_DIR) || (println("Skip BosciaExclusion: $BOSCIA_DIR not found"); continue)
            println("(75 instances per group, exclusion criteria variants)")
            for exclusion_folder in BOSCIA_EXCLUSION_FOLDERS
                merge_boscia_exclusion_group(exclusion_folder, true; verbose)
                merge_boscia_exclusion_group(exclusion_folder, false; verbose)
            end
            println("(ACST / ACSTS exclusion variants, $(length(ALL_KEYS_ACST)) instances each)")
            for crit in ("ACST", "ACSTS")
                for exclusion_folder in BOSCIA_EXCLUSION_FOLDERS
                    merge_boscia_acst_variant(crit, exclusion_folder; verbose)
                end
            end
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
        elseif s == "AGC"
            isdir(BOSCIA_DIR) || (println("Skip AGC (Boscia): $BOSCIA_DIR not found");)
            isdir(SCIPSDP_DIR) || (println("Skip AGC (SCIPSDP): $SCIPSDP_DIR not found");)
            println("(20 instances per group, AGC correlated_connected + independent_disconnected)")
            # Only two AGC setups are used:
            # - correlated_connected
            # - independent_disconnected
            merge_boscia_agc_group(true,  true;  verbose)  # correlated,  connected
            merge_boscia_agc_baseline_group(true, true; verbose)
            merge_boscia_agc_exclusion_group("rank_based_pruning", true, true; verbose)
            for exclusion_folder in BOSCIA_EXCLUSION_FOLDERS
                merge_boscia_agc_exclusion_group(exclusion_folder, true, true; verbose)
            end
            merge_scipsdp_agc_group("oa",  true,  true;  verbose)
            merge_scipsdp_agc_group("bnb", true,  true;  verbose)
            merge_boscia_agc_group(false, false; verbose)  # independent, disconnected
            merge_boscia_agc_baseline_group(false, false; verbose)
            merge_boscia_agc_exclusion_group("rank_based_pruning", false, false; verbose)
            for exclusion_folder in BOSCIA_EXCLUSION_FOLDERS
                merge_boscia_agc_exclusion_group(exclusion_folder, false, false; verbose)
            end
            merge_scipsdp_agc_group("oa",  false, false; verbose)
            merge_scipsdp_agc_group("bnb", false, false; verbose)
        elseif s == "ACSTTrees"
            isdir(BOSCIA_DIR) || (println("Skip ACSTTrees: $BOSCIA_DIR not found"); continue)
            println("($(length(ALL_KEYS_ACST)) instances per criterion × variant)")
            for crit in ("ACST", "ACSTS")
                merge_boscia_acst_default_variant(crit; verbose)
                merge_boscia_acst_baseline_variant(crit; verbose)
                merge_boscia_acst_variant(crit, "rank_based_pruning"; verbose)
                if crit == "ACSTS"
                    merge_boscia_acst_variant(crit, "exclusion_criterion"; verbose)
                end
            end
            if isdir(SCIPSDP_DIR)
                merge_scipsdp_acst_group("oa"; verbose)
                merge_scipsdp_acst_group("bnb"; verbose)
            else
                println("Skip ACST SCIPSDP merge: $SCIPSDP_DIR not found")
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
