#!/usr/bin/env julia

using CSV, DataFrames, Plots, Colors

# --- Config ---
const TIME_LIMIT = 3600.0
const OPTIMAL_CLAIM_TERMINATIONS = Set(["OPTIMAL", "OPTIMALITY_PROVED"])
const REL_TOL = 0.05  # quasi-optimal threshold (5%)

const ODWB_DIR = @__DIR__
const CSV_BASE = joinpath(ODWB_DIR, "csv")
const BOSCIA_DIR = joinpath(CSV_BASE, "Boscia")
const SCIPSDP_DIR = joinpath(CSV_BASE, "SCIPSDP")
const PAJARITO_DIR = joinpath(CSV_BASE, "Pajarito")

const COLOURS_PATH = joinpath(ODWB_DIR, "colours.jl")
isfile(COLOURS_PATH) || error("colours.jl not found at $COLOURS_PATH")
include(COLOURS_PATH)

isfinite_num(x) = x isa Number && isfinite(x)

function is_solved(termination, time)
    t = termination isa String ? termination : string(termination)
    return (t in ("OPTIMAL", "GAPLIMIT", "OPTIMALITY_PROVED")) && (time isa Number) && (time < TIME_LIMIT)
end

function pajarito_feasible_is_false(f)::Bool
    (ismissing(f) || f === nothing) && return false
    if f isa Bool
        return !f
    end
    if f isa Number
        return f == 0
    end
    s = lowercase(strip(string(f)))
    return s in ("false", "0", "no", "f")
end

function rel_tol(best, v)
    max(1e-5, REL_TOL * max(abs(best), abs(v), 1e-12))
end

"""
Mark quasi-optimal OPTIMAL-claim rows as unsolved by setting time=TIME_LIMIT when they are worse than best by > REL_TOL.
Returns count adjusted.
"""
function apply_quasi_optimal_timeout!(df::DataFrame; sense::Symbol)
    keys = [:seed, :dimension, :N, :numberOfParameters]
    all(hasproperty(df, k) for k in keys) || error("missing key columns in df")
    for c in (:scaled_solution, :solver, :termination, :time)
        hasproperty(df, c) || error("missing $c in df")
    end
    n_adj = 0
    g = groupby(df, keys)
    for sub in g
        idxs = parentindices(sub)[1]
        if sense == :min
            best = Inf
            for i in idxs
                v = df[i, :scaled_solution]
                if isfinite_num(v)
                    best = min(best, v)
                end
            end
            isfinite(best) || continue
            for i in idxs
                term = string(df[i, :termination])
                term in OPTIMAL_CLAIM_TERMINATIONS || continue
                v = df[i, :scaled_solution]
                isfinite_num(v) || continue
                if v > best + rel_tol(best, v)
                    df[i, :time] = TIME_LIMIT
                    n_adj += 1
                end
            end
        elseif sense == :max
            best = -Inf
            for i in idxs
                v = df[i, :scaled_solution]
                if isfinite_num(v)
                    best = max(best, v)
                end
            end
            best > -Inf || continue
            for i in idxs
                term = string(df[i, :termination])
                term in OPTIMAL_CLAIM_TERMINATIONS || continue
                v = df[i, :scaled_solution]
                isfinite_num(v) || continue
                if v < best - rel_tol(best, v)
                    df[i, :time] = TIME_LIMIT
                    n_adj += 1
                end
            end
        else
            error("sense must be :min or :max")
        end
    end
    return n_adj
end

function enforce_pajarito_feasible!(df::DataFrame)
    hasproperty(df, :feasible) || return 0
    n = 0
    for i in 1:nrow(df)
        if pajarito_feasible_is_false(df[i, :feasible])
            df[i, :time] = TIME_LIMIT
            n += 1
        end
    end
    return n
end

function step_series_solved(times::Vector{Float64}; time_limit::Float64=TIME_LIMIT, eps::Float64=1e-3)
    t = sort([x for x in times if isfinite(x) && x >= 0.0])
    t = [min(x, time_limit) for x in t]
    # Count solved strictly before time_limit; we still include a final plateau point.
    # NOTE: plots use log-scaled x-axis; avoid 0.0 which would drop the entire series.
    x = vcat(eps, [max(eps, x) for x in t], time_limit)
    y = vcat(0, collect(1:length(t)), length(t))
    return x, y
end

function solver_colors()
    rgb(t) = RGB(t[1], t[2], t[3])
    # Use a stable mapping for the solvers used in tables
    return Dict{String,Any}(
        "Boscia" => rgb(cb_green_lime),
        "SCIPSDP_oa" => rgb(cb_salmon_pink),
        "SCIPSDP_bnb" => rgb(cb_green_sea),
        "ACST (Boscia)" => rgb(cb_green_lime),
        "ACST (rank pruning)" => rgb(cb_blue_green),
        "ACSTS (Boscia)" => rgb(cb_rose),
        "ACSTS (rank pruning)" => rgb(cb_brown),
        "ACSTS (excl.)" => rgb(cb_clay),
        "Pajarito" => rgb(orng),
    )
end

function plot_block(df::DataFrame; solvers::Vector{String}, out_path::String)
    colors = solver_colors()
    plt = plot(
        xlabel="Time (s)",
        ylabel="Solved instances",
        xscale=:log10,
        legend=:bottomright,
        legendfontsize=8,
        grid=true,
        framestyle=:box,
        size=(620, 340),
        title="",  # no titles
    )
    for s in solvers
        sub = df[df.solver .== s, :]
        nrow(sub) == 0 && continue
        solved_times = Float64[]
        for r in eachrow(sub)
            if is_solved(r.termination, r.time)
                push!(solved_times, Float64(r.time))
            end
        end
        x, y = step_series_solved(solved_times)
        plot!(plt, x, y; st=:steppost, label=s, linewidth=2.0, color=get(colors, s, RGB(0,0,0)))
    end
    xlims!(plt, (1e-3, TIME_LIMIT))
    mkpath(dirname(out_path))
    savefig(plt, out_path)
end

function load_eopt(dtype::String)
    boscia = CSV.read(joinpath(BOSCIA_DIR, "boscia_E_optimality_$(dtype)_cont_merged.csv"), DataFrame; delim=';', silencewarnings=true)
    boscia[!, :solver] .= "Boscia"
    oa = CSV.read(joinpath(SCIPSDP_DIR, "scip_sdp_oa_E_optimality_$(dtype)_merged.csv"), DataFrame; delim=',', silencewarnings=true)
    oa[!, :solver] .= "SCIPSDP_oa"
    bnb = CSV.read(joinpath(SCIPSDP_DIR, "scip_sdp_bnb_E_optimality_$(dtype)_merged.csv"), DataFrame; delim=',', silencewarnings=true)
    bnb[!, :solver] .= "SCIPSDP_bnb"
    # Keep only keys we need
    cols = [:seed, :numberOfExperiments, :numberOfParameters, :N, :time, :scaled_solution, :termination, :solver]
    df = vcat(boscia[:, cols], oa[:, cols], bnb[:, cols]; cols=:union)
    df[!, :dimension] = df.numberOfExperiments
    df[!, :numberOfParameters] = df.numberOfParameters
    # quasi-optimal timeout (minimization)
    apply_quasi_optimal_timeout!(df; sense=:min)
    return df
end

function load_agc(dtype::String, conn::String)
    boscia = CSV.read(joinpath(BOSCIA_DIR, "boscia_AGC_optimality_$(dtype)_$(conn)_merged.csv"), DataFrame; delim=';', silencewarnings=true)
    boscia[!, :solver] .= "Boscia"
    oa = CSV.read(joinpath(SCIPSDP_DIR, "scip_sdp_oa_AGC_optimality_$(dtype)_$(conn)_merged.csv"), DataFrame; delim=',', silencewarnings=true)
    oa[!, :solver] .= "SCIPSDP_oa"
    bnb = CSV.read(joinpath(SCIPSDP_DIR, "scip_sdp_bnb_AGC_optimality_$(dtype)_$(conn)_merged.csv"), DataFrame; delim=',', silencewarnings=true)
    bnb[!, :solver] .= "SCIPSDP_bnb"
    cols = [:seed, :numberOfExperiments, :numberOfParameters, :N, :time, :scaled_solution, :termination, :solver]
    df = vcat(boscia[:, cols], oa[:, cols], bnb[:, cols]; cols=:union)
    df[!, :dimension] = df.numberOfExperiments
    apply_quasi_optimal_timeout!(df; sense=:min)
    return df
end

function load_spanning_tree_boscia_unified()
    # ACST/ACSTS use eigenvalue space; aggregated pipeline flips sign before comparisons.
    loaders = [
        ("ACST (Boscia)", joinpath(BOSCIA_DIR, "boscia_ACST_optimality_independent_merged.csv"), ';'),
        ("ACST (rank pruning)", joinpath(BOSCIA_DIR, "boscia_rank_based_pruning_ACST_optimality_independent_merged.csv"), ';'),
        ("ACSTS (Boscia)", joinpath(BOSCIA_DIR, "boscia_ACSTS_optimality_independent_merged.csv"), ';'),
        ("ACSTS (rank pruning)", joinpath(BOSCIA_DIR, "boscia_rank_based_pruning_ACSTS_optimality_independent_merged.csv"), ';'),
        ("ACSTS (excl.)", joinpath(BOSCIA_DIR, "boscia_exclusion_criterion_ACSTS_optimality_independent_merged.csv"), ';'),
    ]
    dfs = DataFrame[]
    for (label, path, d) in loaders
        isfile(path) || continue
        x = CSV.read(path, DataFrame; delim=d, silencewarnings=true)
        x[!, :solver] .= label
        x[!, :dimension] = x.numberOfExperiments
        x[!, :scaled_solution] = -Float64.(x.scaled_solution) # -> λ_min
        push!(dfs, x[:, [:seed,:dimension,:N,:numberOfParameters,:time,:scaled_solution,:termination,:solver]])
    end
    df = vcat(dfs...; cols=:union)
    # NOTE: Boscia-only spanning-tree plots do not cross-check across formulations (same as table policy).
    return df
end

function load_spanning_tree_rank_vs_pajarito()
    b = CSV.read(joinpath(BOSCIA_DIR, "boscia_rank_based_pruning_ACST_optimality_independent_merged.csv"), DataFrame; delim=';', silencewarnings=true)
    b[!, :solver] .= "ACST (rank pruning)"
    b[!, :dimension] = b.numberOfExperiments
    b[!, :scaled_solution] = -Float64.(b.scaled_solution) # -> λ_min
    p = CSV.read(joinpath(PAJARITO_DIR, "pajarito_ACST_optimality_independent_merged.csv"), DataFrame; delim=',', silencewarnings=true)
    p[!, :solver] .= "Pajarito"
    p[!, :dimension] = p.numberOfExperiments
    p[!, :scaled_solution] = -Float64.(p.scaled_solution) # -> λ_min
    if !hasproperty(p,:feasible); p[!, :feasible] = fill(missing, nrow(p)); end
    enforce_pajarito_feasible!(p)
    df = vcat(
        b[:, [:seed,:dimension,:N,:numberOfParameters,:time,:scaled_solution,:termination,:solver]],
        p[:, [:seed,:dimension,:N,:numberOfParameters,:time,:scaled_solution,:termination,:solver,:feasible]];
        cols=:union
    )
    # cross-check only between these two (maximize)
    apply_quasi_optimal_timeout!(df; sense=:max)
    return df
end

function main()
    out_dir = get(ENV, "PLOT_OUT_DIR", "/Users/deborah/Documents/research_projects/Smoothing-in-Boscia/paper")
    out_dir = joinpath(out_dir, "termination_over_time")

    # E-opt
    for dtype in ("independent", "correlated")
        df = load_eopt(dtype)
        plot_block(df; solvers=["Boscia","SCIPSDP_oa","SCIPSDP_bnb"],
            out_path=joinpath(out_dir, "E_opt_$(dtype).pdf"))
    end

    # AGC (only the two setups used in paper)
    plot_block(load_agc("correlated","connected"); solvers=["Boscia","SCIPSDP_oa","SCIPSDP_bnb"],
        out_path=joinpath(out_dir, "AGC_correlated_connected.pdf"))
    plot_block(load_agc("independent","disconnected"); solvers=["Boscia","SCIPSDP_oa","SCIPSDP_bnb"],
        out_path=joinpath(out_dir, "AGC_independent_disconnected.pdf"))

    # Spanning tree
    plot_block(load_spanning_tree_boscia_unified();
        solvers=["ACST (Boscia)","ACST (rank pruning)","ACSTS (Boscia)","ACSTS (rank pruning)","ACSTS (excl.)"],
        out_path=joinpath(out_dir, "spanning_tree_boscia_unified_independent.pdf"))
    plot_block(load_spanning_tree_rank_vs_pajarito();
        solvers=["ACST (rank pruning)","Pajarito"],
        out_path=joinpath(out_dir, "spanning_tree_acst_rank_vs_pajarito_independent.pdf"))

    println("Wrote plots to: ", out_dir)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

