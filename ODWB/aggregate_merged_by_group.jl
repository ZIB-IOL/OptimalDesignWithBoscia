#!/usr/bin/env julia
#=
Build aggregated CSVs combining solvers per data type.
- Default: Boscia + SCIPSDP_oa + SCIPSDP_bnb →
  independent/correlated_by_dimension/N_construction (4 files).
  Use `--all-boscia` to also include exclusion-criterion Boscia variants (merged CSVs required).
f- Smoothing mode (--smoothing): 4 Boscia smoothing regimes (large_mu, small_mu, decay_0.9, decay_0.7) →
  smoothing_independent/correlated_by_dimension/N_construction (4 files).
- AGC mode (--agc): same default four solvers (rank-pruning row omitted if no merged AGC file) →
  agc_correlated_connected_by_dimension.csv and agc_independent_disconnected_by_dimension.csv.
- Spanning tree (`--acst-trees`, or legacy `--acst` / `--acsts`): (1) Boscia-only ACST + ACSTS unified table
  → `spanning_tree_independent_by_dimension.csv`; (2) ACST with rank-based pruning vs **SCIPSDP_oa/SCIPSDP_bnb** on the same
  ACST instances → `spanning_tree_acst_rank_pruning_vs_scipsdp_independent_by_dimension.csv`.
  Default rows in (1): ACST (Boscia), ACST (rank pruning), ACSTS variants; `--all-boscia` adds more ACST/ACSTS columns.
  Cross-solver check: **max** `scaled_solution` in both tables.
- Metrics: geometric mean of time, std w.r.t. geom mean, n_solved, pct_solved, rel_gap geom (unsolved),
  failed_instances, avg_lmo_calls, avg_nodes, etc.

E-optimality and AGC treat `scaled_solution` as a **minimization** objective. ACST / ACSTS treat it as a
**maximization** objective. After merging solvers on the same instance, if a solver reports OPTIMAL (or
OPTIMALITY_PROVED) but its `scaled_solution` is inconsistent with another solver’s feasible value (worse
than the best in the correct direction, within tolerance), we set `time` to TIME_LIMIT and recompute
`solved`.

Run after merge_single_runs_to_csv.jl. Reads merged CSVs from csv/Boscia and csv/SCIPSDP.
=#

using CSV, DataFrames, Statistics

const CSV_BASE = joinpath(@__DIR__, "csv")
const BOSCIA_DIR = joinpath(CSV_BASE, "Boscia")
const SCIPSDP_DIR = joinpath(CSV_BASE, "SCIPSDP")
const PAJARITO_DIR = joinpath(CSV_BASE, "Pajarito")
const TIME_LIMIT = 3600
const BOSCIA_DELIM = ';'
const SCIPSDP_DELIM = ','
const PAJARITO_DELIM = ','

# Prefix folder for rank-based pruning CSVs (same naming as merge_single_runs_to_csv exclusion-style merge).
const BOSCIA_RANK_PRUNING_FOLDER = "rank_based_pruning"
const BOSCIA_RANK_PRUNING_LABEL = "Boscia (rank pruning)"
const ACST_RANK_PRUNING_SOLVER_LABEL = "ACST (rank pruning)"

# Boscia exclusion-criterion variants (folder names) and their display labels.
# These must match the merged CSV filenames produced by `merge_single_runs_to_csv.jl`.
const BOSCIA_EXCLUSION_VARIANTS = [
    ("exclusion_criterion", "Boscia (excl.)"),
    ("exclusion_criterion_random", "Boscia (excl. random)"),
    ("exclusion_criterion_tighter_tol", "Boscia (excl. tighter tol)"),
    ("dual_exclusion_criterion", "Boscia (dual excl.)"),
]

# Terminations that claim a globally optimal solution (used for cross-solver primal check)
const OPTIMAL_CLAIM_TERMINATIONS = ("OPTIMAL", "OPTIMALITY_PROVED")
const QUASI_OPTIMAL_REL_TOL = 0.05

# Solved = reached optimal (or gap limit) within time
function is_solved(termination, time)
    t = termination isa String ? termination : string(termination)
    return t in ("OPTIMAL", "GAPLIMIT", "OPTIMALITY_PROVED") && (time isa Number && time < TIME_LIMIT)
end

"""True iff Pajarito’s CSV `feasible` field explicitly says the post-processed solution is infeasible."""
function pajarito_feasible_is_false(f)::Bool
    (ismissing(f) || f === nothing) && return false
    if f isa Bool
        return !f
    end
    if f isa Number
        return f == 0
    end
    s = lowercase(string(strip(string(f))))
    return s in ("false", "0", "no", "f")
end

"""Same instance key as cross-solver checks: seed, m (dimension), N, n (numberOfParameters)."""
function acst_instance_key_tup(seed, dimension, N, numberOfParameters)
    return (Int(seed), Int(dimension), Int(N), Int(numberOfParameters))
end

"""
`rel_gap` for Pajarito on ACST: compare Pajarito’s feasible scaled objective to Boscia’s certified **lower bound**
`LB = scaled_solution_boscia - dual_gap` (same units as in `opt_design_boscia` / merged CSV), matching Boscia’s
`rel_dual_gap` when the two objectives coincide.
Only defined when `feasible` is not explicitly false and inputs are finite.
"""
function pajarito_acst_rel_gap_vs_boscia_lb(eig_paj, eig_b, dual_gap_b)
    (eig_paj isa Number && isfinite(eig_paj)) || return missing
    (eig_b isa Number && isfinite(eig_b)) || return missing
    (dual_gap_b isa Number && isfinite(dual_gap_b)) || return missing
    # Boscia solves the minimization form; in eigenvalue space:
    #   eig_lb = -primal_min = eig_b
    #   eig_ub = -dual_min = eig_b + dual_gap_b
    eig_ub = eig_b + dual_gap_b
    isfinite(eig_ub) || return missing
    denom = max(abs(eig_ub), 1e-12)
    return max(0.0, (eig_ub - eig_paj) / denom)
end

function apply_pajarito_rel_gap_from_boscia_dual_bound!(df::DataFrame, ref_rank_df::DataFrame)::Int
    hasproperty(ref_rank_df, :scaled_solution) || return 0
    hasproperty(ref_rank_df, :dual_gap) || return 0
    lk = Dict{Tuple{Int,Int,Int,Int},Tuple{Any,Any}}()
    for r in eachrow(ref_rank_df)
        k = acst_instance_key_tup(r.seed, r.dimension, r.N, r.numberOfParameters)
        lk[k] = (r.scaled_solution, r.dual_gap)
    end
    n_set = 0
    for i in 1:nrow(df)
        string(df[i, :solver]) != "Pajarito" && continue
        if hasproperty(df, :feasible) && pajarito_feasible_is_false(df[i, :feasible])
            continue
        end
        k = acst_instance_key_tup(df[i, :seed], df[i, :dimension], df[i, :N], df[i, :numberOfParameters])
        ref = get(lk, k, nothing)
        ref === nothing && continue
        eig_b, dg = ref
        eig_p = df[i, :scaled_solution]
        rg = pajarito_acst_rel_gap_vs_boscia_lb(eig_p, eig_b, dg)
        rg === missing && continue
        df[i, :rel_gap] = rg
        n_set += 1
    end
    return n_set
end

"""
Rows where `feasible` is false: Pajarito already checked the incumbent against the original model and
rejected it. Treat as unsuccessful (time → `TIME_LIMIT`, then `solved` / `failed` refreshed).
Returns the number of rows adjusted.
"""
function apply_pajarito_infeasible_incumbent_penalty!(df::DataFrame)::Int
    hasproperty(df, :feasible) || return 0
    n_adj = 0
    for i in 1:nrow(df)
        pajarito_feasible_is_false(df[i, :feasible]) || continue
        df[i, :time] = Float64(TIME_LIMIT)
        n_adj += 1
    end
    if n_adj > 0
        df[!, :solved] = [is_solved(row.termination, row.time) for row in eachrow(df)]
        df[!, :failed] = [is_failed(row.termination, row.solution, get(row, :solution_source, missing)) for row in eachrow(df)]
    end
    return n_adj
end

function primal_scaled_tol(a, b)
    # Treat as quasi-optimal if worse than best by > 5% (relative).
    max(1e-5, QUASI_OPTIMAL_REL_TOL * max(abs(a), abs(b), 1e-12))
end

"""
Cross-solver consistency on `scaled_solution` per instance (`seed`, `dimension`, `N`, `numberOfParameters`).
`sense=:min` for E-opt / AGC; `sense=:max` for ACST / ACSTS (algebraic connectivity).
OPTIMAL rows strictly worse than the best feasible value are timed out and `solved` is refreshed.
Returns the number of rows adjusted.
"""
function apply_cross_solver_primal_check!(df::DataFrame; sense::Symbol=:min)::Int
    sense in (:min, :max) || error("sense must be :min or :max")
    hasproperty(df, :scaled_solution) || return 0
    keys = [:seed, :dimension, :N, :numberOfParameters]
    all(hasproperty(df, k) for k in keys) || return 0
    if !hasproperty(df, :quasi_optimal)
        df[!, :quasi_optimal] = fill(false, nrow(df))
    end
    n_adj = 0
    g = groupby(df, keys)
    for sub in g
        pi = parentindices(sub)[1]::AbstractVector{<:Integer}
        if sense == :min
            best = Inf
            for idx in pi
                v = df[idx, :scaled_solution]
                if v isa Number && isfinite(v)
                    best = min(best, v)
                end
            end
            best == Inf && continue
            for idx in pi
                term = string(df[idx, :termination])
                term in OPTIMAL_CLAIM_TERMINATIONS || continue
                v = df[idx, :scaled_solution]
                (v isa Number && isfinite(v)) || continue
                tol = primal_scaled_tol(best, v)
                if v > best + tol
                    df[idx, :time] = Float64(TIME_LIMIT)
                    df[idx, :quasi_optimal] = true
                    n_adj += 1
                end
            end
        else
            best = -Inf
            for idx in pi
                v = df[idx, :scaled_solution]
                if v isa Number && isfinite(v)
                    best = max(best, v)
                end
            end
            best == -Inf && continue
            for idx in pi
                term = string(df[idx, :termination])
                term in OPTIMAL_CLAIM_TERMINATIONS || continue
                v = df[idx, :scaled_solution]
                (v isa Number && isfinite(v)) || continue
                tol = primal_scaled_tol(best, v)
                if v < best - tol
                    df[idx, :time] = Float64(TIME_LIMIT)
                    df[idx, :quasi_optimal] = true
                    n_adj += 1
                end
            end
        end
    end
    if n_adj > 0
        df[!, :solved] = [is_solved(row.termination, row.time) for row in eachrow(df)]
    end
    return n_adj
end

"""
Cross-solver check when one method’s rows must stay identical to a precomputed table.
Rows with `solver == reference_solver` are never modified. For `sense=:max`, any other row that
claims OPTIMAL / OPTIMALITY_PROVED but has `scaled_solution` strictly below the reference row’s
value (same instance) is timed out and `solved` is recomputed if anything changed.

Used for the Pajarito comparison table: **ACST (rank pruning)** rows are taken from the unified
Boscia table (after the full multi-formulation check); only **Pajarito** can be downgraded if it
claims optimality but is worse than that reference—which keeps Boscia metrics aligned between tables.
"""
function apply_cross_solver_primal_check_fixed_reference!(
    df::DataFrame,
    reference_solver::AbstractString;
    sense::Symbol=:max,
)::Int
    sense == :max || error("apply_cross_solver_primal_check_fixed_reference! only implements sense=:max")
    hasproperty(df, :scaled_solution) || return 0
    hasproperty(df, :solver) || return 0
    keys = [:seed, :dimension, :N, :numberOfParameters]
    all(hasproperty(df, k) for k in keys) || return 0
    if !hasproperty(df, :quasi_optimal)
        df[!, :quasi_optimal] = fill(false, nrow(df))
    end
    n_adj = 0
    g = groupby(df, keys)
    for sub in g
        pi = parentindices(sub)[1]::AbstractVector{<:Integer}
        ref_idx = [i for i in pi if string(df[i, :solver]) == reference_solver]
        length(ref_idx) == 1 || continue
        ridx = only(ref_idx)
        ref_v = df[ridx, :scaled_solution]
        (ref_v isa Number && isfinite(ref_v)) || continue
        for idx in pi
            string(df[idx, :solver]) == reference_solver && continue
            term = string(df[idx, :termination])
            term in OPTIMAL_CLAIM_TERMINATIONS || continue
            v = df[idx, :scaled_solution]
            (v isa Number && isfinite(v)) || continue
            tol = primal_scaled_tol(ref_v, v)
            if v < ref_v - tol
                df[idx, :time] = Float64(TIME_LIMIT)
                df[idx, :quasi_optimal] = true
                n_adj += 1
            end
        end
    end
    if n_adj > 0
        df[!, :solved] = [is_solved(row.termination, row.time) for row in eachrow(df)]
    end
    return n_adj
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

function load_and_normalize_boscia_baseline(corr::Bool)
    type = corr ? "correlated" : "independent"
    path = joinpath(BOSCIA_DIR, "boscia_baseline_E_optimality_$(type)_merged.csv")
    isfile(path) || return nothing
    df = CSV.read(path, DataFrame; delim=BOSCIA_DELIM, silencewarnings=true)
    n = nrow(df)
    df[!, :solver] = fill("Boscia (baseline)", n)
    df[!, :dimension] = df.numberOfExperiments
    df[!, :rel_gap] = coalesce.(df.rel_dual_gap, Inf)
    df[!, :solved] = [is_solved(row.termination, row.time) for row in eachrow(df)]
    df[!, :failed] = [is_failed(row.termination, row.solution, get(row, :solution_source, missing)) for row in eachrow(df)]
    df[!, :N_construction] = [n_construction_label(row.numberOfParameters, row.N) for row in eachrow(df)]
    df[!, :nodes] = hasproperty(df, :num_nodes) ? df.num_nodes : fill(0, n)
    df[!, :ncalls] = hasproperty(df, :ncalls) ? df.ncalls : fill(0, n)
    df[!, :n_cuts_applied] = fill(0, n)
    df[!, :n_sdp_iters] = fill(0, n)
    return df
end

function load_and_normalize_boscia_e_merged_folder(folder::String, solver_label::String, corr::Bool)
    type = corr ? "correlated" : "independent"
    path = joinpath(BOSCIA_DIR, "boscia_$(folder)_E_optimality_$(type)_merged.csv")
    isfile(path) || return nothing
    df = CSV.read(path, DataFrame; delim=BOSCIA_DELIM, silencewarnings=true)
    n = nrow(df)
    df[!, :solver] = fill(solver_label, n)
    df[!, :dimension] = df.numberOfExperiments
    df[!, :rel_gap] = coalesce.(df.rel_dual_gap, Inf)
    df[!, :solved] = [is_solved(row.termination, row.time) for row in eachrow(df)]
    df[!, :failed] = [is_failed(row.termination, row.solution, get(row, :solution_source, missing)) for row in eachrow(df)]
    df[!, :N_construction] = [n_construction_label(row.numberOfParameters, row.N) for row in eachrow(df)]
    df[!, :nodes] = hasproperty(df, :num_nodes) ? df.num_nodes : fill(0, n)
    df[!, :ncalls] = hasproperty(df, :ncalls) ? df.ncalls : fill(0, n)
    df[!, :n_cuts_applied] = fill(0, n)
    df[!, :n_sdp_iters] = fill(0, n)
    return df
end

function load_and_normalize_boscia_rank_pruning(corr::Bool)
    return load_and_normalize_boscia_e_merged_folder(BOSCIA_RANK_PRUNING_FOLDER, BOSCIA_RANK_PRUNING_LABEL, corr)
end

function load_and_normalize_boscia_exclusion_variant(exclusion_folder::String, corr::Bool)
    solver_label = "Boscia (excl.)"
    for (f, lab) in BOSCIA_EXCLUSION_VARIANTS
        if f == exclusion_folder
            solver_label = lab
            break
        end
    end
    return load_and_normalize_boscia_e_merged_folder(exclusion_folder, solver_label, corr)
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
    path_cont = joinpath(SCIPSDP_DIR, "scip_sdp_$(mode)_E_optimality_$(type)_cont_merged.csv")
    path_plain = joinpath(SCIPSDP_DIR, "scip_sdp_$(mode)_E_optimality_$(type)_merged.csv")
    path = isfile(path_cont) ? path_cont : path_plain
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

function combined_table(corr::Bool; verbose::Bool=false, all_boscia_variants::Bool=false)
    loaders = Any[
        ("Boscia", () -> load_and_normalize_boscia(corr)),
    ]
    if all_boscia_variants
        append!(loaders, [
            ("Boscia (excl.)", () -> load_and_normalize_boscia_exclusion_variant("exclusion_criterion", corr)),
            ("Boscia (excl. random)", () -> load_and_normalize_boscia_exclusion_variant("exclusion_criterion_random", corr)),
            ("Boscia (excl. tighter tol)", () -> load_and_normalize_boscia_exclusion_variant("exclusion_criterion_tighter_tol", corr)),
            ("Boscia (dual excl.)", () -> load_and_normalize_boscia_exclusion_variant("dual_exclusion_criterion", corr)),
        ])
    end
    append!(loaders, [
        ("SCIPSDP_oa", () -> load_and_normalize_scipsdp("oa", corr)),
        ("SCIPSDP_bnb", () -> load_and_normalize_scipsdp("bnb", corr)),
    ])
    dfs = DataFrame[]
    for (_, loader) in loaders
        df = loader()
        df === nothing && continue
        push!(dfs, df)
    end
    isempty(dfs) && return nothing
    common = [:seed, :dimension, :N, :numberOfParameters, :N_construction, :time, :solution, :scaled_solution, :termination, :solver, :rel_gap, :solved, :failed, :nodes, :ncalls, :n_cuts_applied, :n_sdp_iters]
    available = [n for n in common if all(hasproperty(d, n) for d in dfs)]
    df = vcat([df[:, available] for df in dfs]...)
    # E-opt evaluation: ignore rank-deficient construction (N = floor(3n/4)).
    if hasproperty(df, :N_construction)
        df = df[df.N_construction .!= "rank_deficient", :]
    end
    df[!, :quasi_optimal] = fill(false, nrow(df))
    nfix = apply_cross_solver_primal_check!(df; sense=:min)
    if verbose && nfix > 0
        println("  Cross-solver primal check (min scaled_solution): adjusted $nfix rows (time → $(TIME_LIMIT)s, solved recomputed).")
    end
    if hasproperty(df, :scaled_solution)
        select!(df, Not(:scaled_solution))
    end
    return df
end

function combined_table_reduced_vs_baseline(corr::Bool; verbose::Bool=false)
    d_red = load_and_normalize_boscia(corr)  # reduced_spectrum merged under "Boscia"
    d_base = load_and_normalize_boscia_baseline(corr)
    (d_red === nothing || d_base === nothing) && return nothing
    common = [:seed, :dimension, :N, :numberOfParameters, :N_construction, :time, :solution, :scaled_solution, :termination, :solver, :rel_gap, :solved, :failed, :nodes, :ncalls, :n_cuts_applied, :n_sdp_iters]
    df = vcat(d_red[:, common], d_base[:, common])
    if hasproperty(df, :N_construction)
        df = df[df.N_construction .!= "rank_deficient", :]
    end
    df[!, :quasi_optimal] = fill(false, nrow(df))
    nfix = apply_cross_solver_primal_check!(df; sense=:min)
    if verbose && nfix > 0
        println("  Reduced vs baseline primal check (min scaled_solution): adjusted $nfix rows (time → $(TIME_LIMIT)s, solved recomputed).")
    end
    hasproperty(df, :scaled_solution) && select!(df, Not(:scaled_solution))
    return df
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
    df = vcat([df[:, available] for df in dfs]...)
    if hasproperty(df, :N_construction)
        df = df[df.N_construction .!= "rank_deficient", :]
    end
    return df
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

function load_and_normalize_boscia_agc_baseline(corr::Bool, connected::Bool)
    type = corr ? "correlated" : "independent"
    conn = connected ? "connected" : "disconnected"
    path = joinpath(BOSCIA_DIR, "boscia_baseline_AGC_optimality_$(type)_$(conn)_merged.csv")
    isfile(path) || return nothing
    df = CSV.read(path, DataFrame; delim=BOSCIA_DELIM, silencewarnings=true)
    n = nrow(df)
    df[!, :solver] = fill("Boscia (baseline)", n)
    df[!, :dimension] = df.numberOfExperiments
    df[!, :rel_gap] = hasproperty(df, :rel_dual_gap) ? coalesce.(df.rel_dual_gap, Inf) : fill(Inf, n)
    df[!, :solved] = [is_solved(row.termination, row.time) for row in eachrow(df)]
    df[!, :failed] = [is_failed(row.termination, row.solution, get(row, :solution_source, missing)) for row in eachrow(df)]
    df[!, :N_construction] = fill("other", n)
    df[!, :nodes] = hasproperty(df, :num_nodes) ? df.num_nodes : fill(0, n)
    df[!, :ncalls] = hasproperty(df, :ncalls) ? df.ncalls : fill(0, n)
    df[!, :n_cuts_applied] = fill(0, n)
    df[!, :n_sdp_iters] = fill(0, n)
    return df
end

function load_and_normalize_boscia_agc_merged_folder(folder::String, solver_label::String, corr::Bool, connected::Bool)
    type = corr ? "correlated" : "independent"
    conn = connected ? "connected" : "disconnected"
    path = joinpath(BOSCIA_DIR, "boscia_$(folder)_AGC_optimality_$(type)_$(conn)_merged.csv")
    isfile(path) || return nothing
    df = CSV.read(path, DataFrame; delim=BOSCIA_DELIM, silencewarnings=true)
    n = nrow(df)
    df[!, :solver] = fill(solver_label, n)
    df[!, :dimension] = df.numberOfExperiments
    df[!, :rel_gap] = hasproperty(df, :rel_dual_gap) ? coalesce.(df.rel_dual_gap, Inf) : fill(Inf, n)
    df[!, :solved] = [is_solved(row.termination, row.time) for row in eachrow(df)]
    df[!, :failed] = [is_failed(row.termination, row.solution, get(row, :solution_source, missing)) for row in eachrow(df)]
    df[!, :N_construction] = fill("other", n)
    df[!, :nodes] = hasproperty(df, :num_nodes) ? df.num_nodes : fill(0, n)
    df[!, :ncalls] = hasproperty(df, :ncalls) ? df.ncalls : fill(0, n)
    df[!, :n_cuts_applied] = fill(0, n)
    df[!, :n_sdp_iters] = fill(0, n)
    return df
end

function load_and_normalize_boscia_agc_rank_pruning(corr::Bool, connected::Bool)
    return load_and_normalize_boscia_agc_merged_folder(BOSCIA_RANK_PRUNING_FOLDER, BOSCIA_RANK_PRUNING_LABEL, corr, connected)
end

function load_and_normalize_boscia_agc_exclusion_variant(exclusion_folder::String, corr::Bool, connected::Bool)
    solver_label = "Boscia (excl.)"
    for (f, lab) in BOSCIA_EXCLUSION_VARIANTS
        if f == exclusion_folder
            solver_label = lab
            break
        end
    end
    return load_and_normalize_boscia_agc_merged_folder(exclusion_folder, solver_label, corr, connected)
end

function load_and_normalize_boscia_agc_exclusion(corr::Bool, connected::Bool)
    return load_and_normalize_boscia_agc_exclusion_variant("exclusion_criterion", corr, connected)
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

function combined_table_agc(corr::Bool, connected::Bool; verbose::Bool=false, all_boscia_variants::Bool=false)
    loaders = Any[
        ("Boscia", () -> load_and_normalize_boscia_agc(corr, connected)),
    ]
    if all_boscia_variants
        append!(loaders, [
            ("Boscia (excl.)", () -> load_and_normalize_boscia_agc_exclusion_variant("exclusion_criterion", corr, connected)),
            ("Boscia (excl. random)", () -> load_and_normalize_boscia_agc_exclusion_variant("exclusion_criterion_random", corr, connected)),
            ("Boscia (excl. tighter tol)", () -> load_and_normalize_boscia_agc_exclusion_variant("exclusion_criterion_tighter_tol", corr, connected)),
            ("Boscia (dual excl.)", () -> load_and_normalize_boscia_agc_exclusion_variant("dual_exclusion_criterion", corr, connected)),
        ])
    end
    append!(loaders, [
        ("SCIPSDP_oa", () -> load_and_normalize_scipsdp_agc("oa", corr, connected)),
        ("SCIPSDP_bnb", () -> load_and_normalize_scipsdp_agc("bnb", corr, connected)),
    ])
    dfs = DataFrame[]
    for (_, loader) in loaders
        df = loader()
        df === nothing && continue
        push!(dfs, df)
    end
    isempty(dfs) && return nothing
    common = [:seed, :dimension, :N, :numberOfParameters, :N_construction, :time, :solution, :scaled_solution, :termination, :solver, :rel_gap, :solved, :failed, :nodes, :ncalls, :n_cuts_applied, :n_sdp_iters]
    available = [n for n in common if all(hasproperty(d, n) for d in dfs)]
    df = vcat([df[:, available] for df in dfs]...)
    df[!, :quasi_optimal] = fill(false, nrow(df))
    # Evaluation policy: if a solver claims OPTIMAL/OPTIMALITY_PROVED but reports a worse objective than
    # another solver on the same instance (beyond tolerance), treat it as suboptimal: time -> TIME_LIMIT
    # and do not count it as solved.
    nfix = apply_cross_solver_primal_check!(df; sense=:min)
    if verbose && nfix > 0
        println("  Cross-solver primal check (min scaled_solution): adjusted $nfix rows (time → $(TIME_LIMIT)s, solved recomputed).")
    end
    if hasproperty(df, :scaled_solution)
        select!(df, Not(:scaled_solution))
    end
    return df
end

function combined_table_agc_reduced_vs_baseline(corr::Bool, connected::Bool; verbose::Bool=false)
    d_red = load_and_normalize_boscia_agc(corr, connected)
    d_base = load_and_normalize_boscia_agc_baseline(corr, connected)
    (d_red === nothing || d_base === nothing) && return nothing
    common = [:seed, :dimension, :N, :numberOfParameters, :N_construction, :time, :solution, :scaled_solution, :termination, :solver, :rel_gap, :solved, :failed, :nodes, :ncalls, :n_cuts_applied, :n_sdp_iters]
    df = vcat(d_red[:, common], d_base[:, common])
    df[!, :quasi_optimal] = fill(false, nrow(df))
    nfix = apply_cross_solver_primal_check!(df; sense=:min)
    if verbose && nfix > 0
        println("  AGC reduced vs baseline primal check (min scaled_solution): adjusted $nfix rows.")
    end
    hasproperty(df, :scaled_solution) && select!(df, Not(:scaled_solution))
    return df
end

function load_and_normalize_boscia_acst_variant(criterion::String, folder::Union{Nothing,String}, solver_label::String)
    path = if folder === nothing
        joinpath(BOSCIA_DIR, "boscia_$(criterion)_optimality_independent_merged.csv")
    else
        joinpath(BOSCIA_DIR, "boscia_$(folder)_$(criterion)_optimality_independent_merged.csv")
    end
    isfile(path) || return nothing
    df = CSV.read(path, DataFrame; delim=BOSCIA_DELIM, silencewarnings=true)
    n = nrow(df)
    df[!, :solver] = fill(solver_label, n)
    df[!, :dimension] = df.numberOfExperiments
    # ACST/ACSTS are exported in the minimization form (-λ_min). Convert to λ_min for comparisons/tables.
    if hasproperty(df, :scaled_solution)
        df[!, :scaled_solution] = -Float64.(df.scaled_solution)
    end
    if !hasproperty(df, :dual_gap)
        df[!, :dual_gap] = fill(missing, n)
    end
    df[!, :rel_gap] = hasproperty(df, :rel_dual_gap) ? coalesce.(df.rel_dual_gap, Inf) : fill(Inf, n)
    df[!, :solved] = [is_solved(row.termination, row.time) for row in eachrow(df)]
    df[!, :failed] = [is_failed(row.termination, row.solution, get(row, :solution_source, missing)) for row in eachrow(df)]
    df[!, :N_construction] = fill("other", n)
    df[!, :nodes] = hasproperty(df, :num_nodes) ? df.num_nodes : fill(0, n)
    df[!, :ncalls] = hasproperty(df, :ncalls) ? df.ncalls : fill(0, n)
    df[!, :n_cuts_applied] = fill(0, n)
    df[!, :n_sdp_iters] = fill(0, n)
    return df
end

function load_and_normalize_pajarito_acst_merged()
    path = joinpath(PAJARITO_DIR, "pajarito_ACST_optimality_independent_merged.csv")
    isfile(path) || return nothing
    df = CSV.read(path, DataFrame; delim=PAJARITO_DELIM, silencewarnings=true)
    n = nrow(df)
    df[!, :solver] = fill("Pajarito", n)
    df[!, :dimension] = df.numberOfExperiments
    # Pajarito ACST CSV uses the same convention (-λ_min). Convert to λ_min for comparisons/tables.
    if hasproperty(df, :scaled_solution)
        df[!, :scaled_solution] = -Float64.(df.scaled_solution)
    end
    df[!, :rel_gap] = fill(Inf, n)
    df[!, :dual_gap] = fill(missing, n)
    if !hasproperty(df, :feasible)
        df[!, :feasible] = fill(missing, n)
    end
    df[!, :solved] = [is_solved(row.termination, row.time) for row in eachrow(df)]
    df[!, :failed] = [is_failed(row.termination, row.solution, get(row, :solution_source, missing)) for row in eachrow(df)]
    df[!, :N_construction] = fill("other", n)
    df[!, :nodes] = hasproperty(df, :numberIterations) ? df.numberIterations : fill(0, n)
    df[!, :ncalls] = fill(0, n)
    df[!, :n_cuts_applied] = hasproperty(df, :numberCuts) ? [Float64(coalesce(c, 0)) for c in df.numberCuts] : fill(0.0, n)
    df[!, :n_sdp_iters] = fill(0, n)
    apply_pajarito_infeasible_incumbent_penalty!(df)
    return df
end

function load_and_normalize_scipsdp_acst_merged(mode::String)
    path = joinpath(SCIPSDP_DIR, "scip_sdp_$(mode)_ACST_optimality_independent_merged.csv")
    isfile(path) || return nothing
    df = CSV.read(path, DataFrame; delim=SCIPSDP_DELIM, silencewarnings=true)
    n = nrow(df)
    df[!, :solver] = fill("SCIPSDP_$(mode)", n)
    df[!, :dimension] = df.numberOfExperiments
    # ACST SCIPSDP CSVs use the minimization form (-λ_min). Convert to λ_min for comparisons/tables.
    if hasproperty(df, :scaled_solution)
        df[!, :scaled_solution] = -Float64.(df.scaled_solution)
    end
    df[!, :rel_gap] = hasproperty(df, :rel_gap) ? coalesce.(df.rel_gap, Inf) : fill(Inf, n)
    if !hasproperty(df, :dual_gap)
        df[!, :dual_gap] = fill(missing, n)
    end
    if !hasproperty(df, :feasible)
        df[!, :feasible] = fill(missing, n)
    end
    df[!, :solved] = [is_solved(row.termination, row.time) for row in eachrow(df)]
    df[!, :failed] = [is_failed(row.termination, row.solution, get(row, :solution_source, missing)) for row in eachrow(df)]
    df[!, :N_construction] = fill("other", n)
    df[!, :nodes] = hasproperty(df, :n_nodes) ? df.n_nodes : fill(0, n)
    df[!, :ncalls] = fill(0, n)
    df[!, :n_cuts_applied] = (mode == "oa" && hasproperty(df, :n_cuts_applied)) ? df.n_cuts_applied : fill(0, n)
    df[!, :n_sdp_iters] = (mode == "bnb" && hasproperty(df, :n_sdp_iters)) ? coalesce.(df.n_sdp_iters, 0) : fill(0, n)
    return df
end

function combined_table_spanning_tree_unified(;
    verbose::Bool=false,
    all_boscia_variants::Bool=false,
    drop_scaled_solution::Bool=true,
)
    loaders = Any[]
    # ACST (matrix / hyperplane-aware formulation)
    push!(loaders, () -> load_and_normalize_boscia_acst_variant("ACST", nothing, "ACST (Boscia)"))
    push!(loaders, () -> load_and_normalize_boscia_acst_variant("ACST", "rank_based_pruning", "ACST (rank pruning)"))
    if all_boscia_variants
        for (f, lab) in BOSCIA_EXCLUSION_VARIANTS
            acst_lab = replace(lab, "Boscia" => "ACST", count=1)
            push!(loaders, () -> load_and_normalize_boscia_acst_variant("ACST", f, acst_lab))
        end
    end
    # ACSTS (probability / simple LMO formulation)
    push!(loaders, () -> load_and_normalize_boscia_acst_variant("ACSTS", nothing, "ACSTS (Boscia)"))
    push!(loaders, () -> load_and_normalize_boscia_acst_variant("ACSTS", "rank_based_pruning", "ACSTS (rank pruning)"))
    push!(loaders, () -> load_and_normalize_boscia_acst_variant("ACSTS", "exclusion_criterion", "ACSTS (excl.)"))
    if all_boscia_variants
        for (f, lab) in BOSCIA_EXCLUSION_VARIANTS
            f == "exclusion_criterion" && continue
            acsts_lab = replace(lab, "Boscia" => "ACSTS", count=1)
            push!(loaders, () -> load_and_normalize_boscia_acst_variant("ACSTS", f, acsts_lab))
        end
    end
    dfs = DataFrame[]
    for loader in loaders
        df = loader()
        df === nothing && continue
        push!(dfs, df)
    end
    isempty(dfs) && return nothing
    common = [:seed, :dimension, :N, :numberOfParameters, :N_construction, :time, :solution, :scaled_solution, :dual_gap, :termination, :solver, :rel_gap, :solved, :failed, :nodes, :ncalls, :n_cuts_applied, :n_sdp_iters]
    available = [nm for nm in common if all(hasproperty(d, nm) for d in dfs)]
    df = vcat([df[:, available] for df in dfs]...)
    # NOTE: we intentionally do NOT downgrade Boscia spanning-tree variants based on cross-method objective
    # comparisons. Different formulations/variants can legitimately return different values; for the Boscia-only
    # table we report solver status as-is. The strict invalidation logic is applied only in the Pajarito
    # comparison table (Pajarito vs frozen ACST rank-pruning reference, plus Pajarito's own `feasible` flag).
    if drop_scaled_solution && hasproperty(df, :scaled_solution)
        select!(df, Not(:scaled_solution))
    end
    return df
end

function combined_table_spanning_tree_acst_reduced_vs_baseline(; verbose::Bool=false)
    d_red = load_and_normalize_boscia_acst_variant("ACST", nothing, "ACST (Boscia)")
    d_base = load_and_normalize_boscia_acst_variant("ACST", "baseline", "ACST (baseline)")
    (d_red === nothing || d_base === nothing) && return nothing
    common = [:seed, :dimension, :N, :numberOfParameters, :N_construction, :time, :solution, :scaled_solution, :dual_gap, :termination, :solver, :rel_gap, :solved, :failed, :nodes, :ncalls, :n_cuts_applied, :n_sdp_iters]
    df = vcat(d_red[:, common], d_base[:, common])
    df[!, :quasi_optimal] = fill(false, nrow(df))
    nfix = apply_cross_solver_primal_check!(df; sense=:max)
    if verbose && nfix > 0
        println("  ACST reduced vs baseline primal check (max scaled_solution): adjusted $nfix rows.")
    end
    if hasproperty(df, :scaled_solution)
        select!(df, Not(:scaled_solution))
    end
    hasproperty(df, :dual_gap) && select!(df, Not(:dual_gap))
    return df
end

"""
ACST rank pruning vs Pajarito with frozen Boscia reference.

The ACST (rank pruning) rows are copied from the already cross-checked unified Boscia table, so metrics
stay identical across both outputs. Pajarito rows with `feasible == false` are invalidated when loading.
Remaining Pajarito rows are validated against that reference: if Pajarito claims OPTIMAL/OPTIMALITY_PROVED
but has a worse `scaled_solution`, its run is invalidated (time → limit).

For Pajarito, `rel_gap` uses Boscia ACST (rank pruning) `LB = scaled_solution - dual_gap` on the same instance;
`(scaled_pajarito - LB) / max(|scaled_pajarito|, 1e-12)` when `feasible` is not explicitly false (aligns with Boscia’s
`rel_dual_gap` if objectives coincide).
"""
function combined_table_spanning_tree_acst_rank_pruning_vs_scipsdp(;
    verbose::Bool=false,
    unified_df_with_scaled::Union{Nothing,DataFrame}=nothing,
)
    df_u = unified_df_with_scaled
    if df_u === nothing
        df_u = combined_table_spanning_tree_unified(; verbose=false, drop_scaled_solution=false)
    end
    df_u === nothing && return nothing
    hasproperty(df_u, :scaled_solution) || error("unified_df_with_scaled must include :scaled_solution")

    d1 = copy(df_u[df_u.solver .== ACST_RANK_PRUNING_SOLVER_LABEL, :])
    nrow(d1) == 0 && return nothing
    if !hasproperty(d1, :feasible)
        d1[!, :feasible] = fill(missing, nrow(d1))
    end
    d_oa = load_and_normalize_scipsdp_acst_merged("oa")
    d_bnb = load_and_normalize_scipsdp_acst_merged("bnb")
    scip_parts = DataFrame[]
    d_oa !== nothing && push!(scip_parts, d_oa)
    d_bnb !== nothing && push!(scip_parts, d_bnb)
    isempty(scip_parts) && return nothing
    d2 = vcat(scip_parts...; cols=:union)

    common = [:seed, :dimension, :N, :numberOfParameters, :N_construction, :time, :solution, :scaled_solution, :dual_gap, :termination, :solver, :rel_gap, :solved, :failed, :nodes, :ncalls, :n_cuts_applied, :n_sdp_iters, :feasible]
    available = [nm for nm in common if hasproperty(d1, nm) && hasproperty(d2, nm)]
    df = vcat(d1[:, available], d2[:, available]; cols=:orderequal)
    df[!, :quasi_optimal] = fill(false, nrow(df))

    nfix = apply_cross_solver_primal_check_fixed_reference!(df, ACST_RANK_PRUNING_SOLVER_LABEL; sense=:max)
    if verbose && nfix > 0
        println("  SCIPSDP optimality guard vs ACST rank-pruning reference: invalidated $nfix rows.")
    end
    if hasproperty(df, :scaled_solution)
        select!(df, Not(:scaled_solution))
    end
    for col in (:dual_gap, :feasible)
        hasproperty(df, col) && select!(df, Not(col))
    end
    return df
end

function run_aggregation_spanning_trees(; out_dir=nothing, all_boscia_variants=false, verbose=true)
    out_dir = something(out_dir, joinpath(CSV_BASE, "aggregated"))
    mkpath(out_dir)
    if verbose
        println("Aggregating spanning-tree results (Boscia ACST+ACSTS; plus ACST rank pruning vs SCIPSDP).")
        println("Output directory: $out_dir")
    end
    verbose && println("\n--- spanning_tree_independent (Boscia formulations) ---")
    df_with_scaled = combined_table_spanning_tree_unified(; verbose, all_boscia_variants, drop_scaled_solution=false)
    df = isnothing(df_with_scaled) ? nothing : copy(df_with_scaled)
    if df !== nothing && hasproperty(df, :scaled_solution)
        select!(df, Not(:scaled_solution))
    end
    if df === nothing
        verbose && println("  No Boscia spanning-tree data, skipping first table.")
    else
        verbose && println("  Combined rows: $(nrow(df)), methods: $(unique(df.solver))")
        by_dim = vcat(aggregate_by(df, :dimension), aggregate_overall(df, :dimension); cols=:union)
        out_dim = joinpath(out_dir, "spanning_tree_independent_by_dimension.csv")
        CSV.write(out_dim, by_dim)
        verbose && println("  Wrote $out_dim ($(nrow(by_dim)) rows)")
    end
    verbose && println("\n--- spanning_tree_acst_rank_pruning_vs_scipsdp ---")
    df2 = combined_table_spanning_tree_acst_rank_pruning_vs_scipsdp(; verbose, unified_df_with_scaled=df_with_scaled)
    if df2 !== nothing
        verbose && println("  Combined rows: $(nrow(df2)), methods: $(unique(df2.solver))")
        by_dim2 = vcat(aggregate_by(df2, :dimension), aggregate_overall(df2, :dimension); cols=:union)
        out_paj = joinpath(out_dir, "spanning_tree_acst_rank_pruning_vs_scipsdp_independent_by_dimension.csv")
        CSV.write(out_paj, by_dim2)
        verbose && println("  Wrote $out_paj ($(nrow(by_dim2)) rows)")
    else
        verbose && println("\n--- spanning_tree_acst_rank_pruning_vs_scipsdp: skipped (no rank-pruning and/or SCIPSDP merged CSV) ---")
    end
    verbose && println("\n--- spanning_tree_acst_reduced_vs_baseline ---")
    df3 = combined_table_spanning_tree_acst_reduced_vs_baseline(; verbose)
    if df3 !== nothing
        by_dim3 = vcat(aggregate_by(df3, :dimension), aggregate_overall(df3, :dimension); cols=:union)
        out_cmp = joinpath(out_dir, "spanning_tree_acst_reduced_vs_baseline_independent_by_dimension.csv")
        CSV.write(out_cmp, by_dim3)
        verbose && println("  Wrote $out_cmp ($(nrow(by_dim3)) rows)")
    elseif verbose
        println("  skipped (missing reduced and/or baseline ACST merged CSV).")
    end
    if verbose
        println("\nDone. Outputs: spanning_tree_independent_by_dimension.csv (if Boscia data), " *
            "spanning_tree_acst_rank_pruning_vs_scipsdp_independent_by_dimension.csv (if all methods load), " *
            "spanning_tree_acst_reduced_vs_baseline_independent_by_dimension.csv (if both variants load).")
    end
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
        quasi_opt = hasproperty(sdf, :quasi_optimal) ? count(sdf.quasi_optimal) : 0
        # Averages over solved instances only; 0 when not applicable for that solver
        solved_idx = findall(solved)
        n_sol = length(solved_idx)
        is_boscia_like = startswith(solver, "Boscia") || solver in SMOOTHING_REGIMES || startswith(solver, "ACST")
        avg_lmo_calls = (is_boscia_like && n_sol > 0) ? round(sum(sdf.ncalls[solved_idx]) / n_sol; digits=2) : 0.0
        avg_nodes = n_sol > 0 ? round(sum(sdf.nodes[solved_idx]) / n_sol; digits=2) : 0.0
        avg_cuts = ((solver == "SCIPSDP_oa" || solver == "Pajarito") && n_sol > 0) ? round(sum(skipmissing(sdf.n_cuts_applied[solved_idx])) / n_sol; digits=2) : 0.0
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
            :quasi_optimal => quasi_opt,
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
    order = [group_col, :solver, :n_instances, :n_solved, :pct_solved, :time_geom_mean, :time_std_wrt_geom, :rel_gap_geom_mean_unsolved, :failed_instances, :quasi_optimal, :avg_lmo_calls, :avg_nodes, :avg_cuts, :avg_sdp_iters]
    nms = names(out)
    # Match both Symbol and String column names (DataFrame(rows) may use either)
    cols = [c for c in order if c in nms || string(c) in nms]
    if isempty(cols)
        return out
    end
    idx = [c in nms ? c : string(c) for c in order if c in nms || string(c) in nms]
    return out[:, idx]
end

"""
Overall aggregation across *all* instances for each solver, emitted as an extra row in by-dimension CSVs.
We encode the overall column as `dimension = -1` (rendered as `all` in LaTeX).
"""
function aggregate_overall(df::DataFrame, group_col::Symbol; overall_value=-1)
    tmp = copy(df)
    tmp[!, group_col] = fill(overall_value, nrow(tmp))
    return aggregate_by(tmp, group_col)
end

function run_aggregation(; out_dir=nothing, smoothing=false, all_boscia_variants=false, verbose=true)
    out_dir = something(out_dir, joinpath(CSV_BASE, "aggregated"))
    mkpath(out_dir)
    prefix = smoothing ? "smoothing_" : ""
    if verbose
        if smoothing
            println("Aggregating Boscia smoothing regimes (4) by data type and dimension/N.")
        elseif all_boscia_variants
            println("Aggregating merged results (Boscia + rank pruning + all exclusion Boscia variants + SCIPSDP) by data type and dimension/N.")
        else
            println("Aggregating merged results (Boscia + rank pruning + SCIPSDP) by data type and dimension/N.")
        end
        println("Output directory: $out_dir")
    end
    for corr in (false, true)
        data_type = corr ? "correlated" : "independent"
        verbose && println("\n--- $(prefix)$data_type ---")
        df = smoothing ? combined_table_smoothing(corr) : combined_table(corr; verbose, all_boscia_variants)
        if df === nothing
            verbose && println("  No data, skipping.")
            continue
        end
        if verbose
            println("  Combined rows: $(nrow(df)), solvers/regimes: $(unique(df.solver))")
        end
        by_dim = vcat(aggregate_by(df, :dimension), aggregate_overall(df, :dimension); cols=:union)
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

        if !smoothing
            df_cmp = combined_table_reduced_vs_baseline(corr; verbose)
            if df_cmp !== nothing
                by_dim_cmp = vcat(aggregate_by(df_cmp, :dimension), aggregate_overall(df_cmp, :dimension); cols=:union)
                out_cmp = joinpath(out_dir, "$(data_type)_reduced_vs_baseline_by_dimension.csv")
                CSV.write(out_cmp, by_dim_cmp)
                verbose && println("  Wrote $out_cmp ($(nrow(by_dim_cmp)) rows)")
            elseif verbose
                println("  Skip reduced-vs-baseline table for $data_type (missing merged baseline/reduced CSV).")
            end
        end
    end
    if verbose
        println("\nDone. Outputs: $(prefix)*_by_dimension.csv, $(prefix)*_by_N_construction.csv")
    end
end

function run_aggregation_agc(; out_dir=nothing, all_boscia_variants=false, verbose=true)
    out_dir = something(out_dir, joinpath(CSV_BASE, "aggregated"))
    mkpath(out_dir)
    if verbose
        msg = all_boscia_variants ?
            "Aggregating AGC results (Boscia + rank pruning + exclusion variants + SCIPSDP) by dimension." :
            "Aggregating AGC results (Boscia + rank pruning + SCIPSDP) by dimension."
        println(msg)
        println("Output directory: $out_dir")
    end
    # Only two AGC setups are used: correlated_connected and independent_disconnected
    setups = [
        (true,  true,  "correlated_connected"),
        (false, false, "independent_disconnected"),
    ]
    for (corr, connected, tag) in setups
        verbose && println("\n--- agc_$tag ---")
        df = combined_table_agc(corr, connected; verbose, all_boscia_variants)
        if df === nothing
            verbose && println("  No AGC data, skipping.")
            continue
        end
        if verbose
            println("  Combined rows: $(nrow(df)), solvers: $(unique(df.solver))")
        end
        by_dim = vcat(aggregate_by(df, :dimension), aggregate_overall(df, :dimension); cols=:union)
        out_dim = joinpath(out_dir, "agc_$(tag)_by_dimension.csv")
        CSV.write(out_dim, by_dim)
        if verbose
            println("  Wrote $out_dim ($(nrow(by_dim)) rows)")
        end
        df_cmp = combined_table_agc_reduced_vs_baseline(corr, connected; verbose)
        if df_cmp !== nothing
            by_dim_cmp = vcat(aggregate_by(df_cmp, :dimension), aggregate_overall(df_cmp, :dimension); cols=:union)
            out_cmp = joinpath(out_dir, "agc_$(tag)_reduced_vs_baseline_by_dimension.csv")
            CSV.write(out_cmp, by_dim_cmp)
            verbose && println("  Wrote $out_cmp ($(nrow(by_dim_cmp)) rows)")
        elseif verbose
            println("  Skip reduced-vs-baseline AGC for $tag (missing merged baseline/reduced CSV).")
        end
    end
    if verbose
        println("\nDone. Outputs: agc_*_by_dimension.csv")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    smoothing = "--smoothing" in ARGS
    agc = "--agc" in ARGS
    spanning = "--acst-trees" in ARGS || "--acst" in ARGS || "--acsts" in ARGS
    all_boscia = "--all-boscia" in ARGS
    filter!(x -> x ∉ ("--smoothing", "--agc", "--all-boscia", "--acst-trees", "--acst", "--acsts"), ARGS)
    if agc
        run_aggregation_agc(; all_boscia_variants=all_boscia)
    elseif spanning
        run_aggregation_spanning_trees(; all_boscia_variants=all_boscia)
    else
        run_aggregation(; smoothing, all_boscia_variants=all_boscia)
    end
end
