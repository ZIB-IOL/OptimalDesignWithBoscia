#!/usr/bin/env julia
#=
Generate LaTeX tables from aggregated CSVs (by_dimension and by_N_construction).
- Default solver order matches aggregate_merged_by_group.jl: Boscia, SCIPSDP_oa, SCIPSDP_bnb
  (only columns present in the CSV are emitted).
- `--acst-trees` (or legacy `--acst` / `--acsts`): `spanning_tree_independent_by_dimension.tex` from the Boscia-only
  unified CSV, and `spanning_tree_acst_rank_pruning_vs_scipsdp_independent_by_dimension.tex` when that aggregate exists.
  `--all-boscia` only affects the first table’s solver order.
- Use `--all-boscia` to also expect exclusion-criterion Boscia variants in the CSV order (E / AGC / spanning tree).
- Rows = dimension or N_construction.
- Option to hide columns via --hide (comma-separated list).

Hideable column names: pct_solved, time_geom_mean, time_std_wrt_geom,
  rel_gap_geom_mean_unsolved, failed_instances, avg_lmo_calls, avg_nodes,
  avg_cuts, avg_sdp_iters.

Usage:
  julia aggregated_to_tex.jl [...] [--smoothing] [--agc] [--acst-trees] [--acst] [--acsts] [--all-boscia]
  julia aggregated_to_tex.jl --smoothing   # Boscia smoothing regimes (4) → smoothing_*.tex
  julia aggregated_to_tex.jl --agc         # AGC setups (2) → agc_*.tex
  julia aggregated_to_tex.jl --acst-trees  # Boscia ACST+ACSTS + ACST rank pr. vs SCIPSDP → two .tex files when CSVs exist
  julia aggregated_to_tex.jl --hide avg_nodes time_std_wrt_geom

All arguments after --hide (until the next --) are treated as column names to hide; commas are optional.

Output: writes .tex files into the paper repo (or --out DIR) and prints required preamble.
=#

using CSV, DataFrames, Printf

const AGG_DIR = joinpath(@__DIR__, "csv", "aggregated")
const TEX_OUT_DIR = "/Users/deborah/Documents/research_projects/Smoothing-in-Boscia/paper"
# Default: same four-way comparison as aggregate_merged_by_group.jl (no --all-boscia).
const SOLVERS_DEFAULT = [
    "Boscia",
    "SCIPSDP_oa",
    "SCIPSDP_bnb",
]
const SOLVERS_REDUCED_VS_BASELINE = [
    "Boscia",
    "Boscia (baseline)",
]
const SOLVERS_WITH_EXCLUSIONS = [
    "Boscia",
    "Boscia (excl.)",
    "Boscia (excl. random)",
    "Boscia (excl. tighter tol)",
    "Boscia (dual excl.)",
    "SCIPSDP_oa",
    "SCIPSDP_bnb",
]
# Unified spanning-tree table (same order as aggregate `combined_table_spanning_tree_unified`).
const SOLVERS_TEX_SPANNING_TREE = [
    "ACST (Boscia)",
    "ACST (rank pruning)",
    "ACSTS (Boscia)",
    "ACSTS (rank pruning)",
    "ACSTS (excl.)",
]
const SOLVERS_TEX_SPANNING_TREE_ALL = [
    "ACST (Boscia)",
    "ACST (rank pruning)",
    "ACST (excl.)",
    "ACST (excl. random)",
    "ACST (excl. tighter tol)",
    "ACST (dual excl.)",
    "ACSTS (Boscia)",
    "ACSTS (rank pruning)",
    "ACSTS (excl.)",
    "ACSTS (excl. random)",
    "ACSTS (excl. tighter tol)",
    "ACSTS (dual excl.)",
]
const SOLVERS_TEX_ACST_RANK_VS_SCIPSDP = ["ACST (Boscia)", "SCIPSDP_oa", "SCIPSDP_bnb"]
const SOLVERS_TEX_ACST_REDUCED_VS_BASELINE = ["ACST (Boscia)", "ACST (baseline)"]
const SOLVER_LABELS = Dict(
    "Boscia" => "Boscia",
    "Boscia (excl.)" => "Boscia (excl.)",
    "Boscia (excl. random)" => "Boscia (excl. random)",
    "Boscia (excl. tighter tol)" => "Boscia (excl. tighter tol)",
    "Boscia (dual excl.)" => "Boscia (dual excl.)",
    "Boscia (baseline)" => "Boscia (baseline)",
    "ACST (Boscia)" => "Boscia",
    "ACST (baseline)" => "ACST (baseline)",
    "ACST (rank pruning)" => "ACST (rank pr.)",
    "ACST (excl.)" => "ACST (excl.)",
    "ACST (excl. random)" => "ACST (excl. rand.)",
    "ACST (excl. tighter tol)" => "ACST (excl. tight)",
    "ACST (dual excl.)" => "ACST (dual excl.)",
    "ACSTS (Boscia)" => "ACSTS (def.)",
    "ACSTS (rank pruning)" => "ACSTS (rank pr.)",
    "ACSTS (excl.)" => "ACSTS (excl.)",
    "ACSTS (excl. random)" => "ACSTS (excl. rand.)",
    "ACSTS (excl. tighter tol)" => "ACSTS (excl. tight)",
    "ACSTS (dual excl.)" => "ACSTS (dual excl.)",
    "SCIPSDP_oa" => "SCIPSDP (OA)",
    "SCIPSDP_bnb" => "SCIPSDP (B\\&B)",
)
const SMOOTHING_REGIMES = ["large_mu", "small_mu", "decay_0.9", "decay_0.7"]
const SMOOTHING_LABELS = Dict(
    "large_mu" => "Large \$\\mu\$ (decay=1)",
    "small_mu" => "Small \$\\mu\$ (decay=1)",
    "decay_0.9" => "Decay 0.9",
    "decay_0.7" => "Decay 0.7",
)
const DEFAULT_HIDDEN_METRICS = Set([
    "time_std_wrt_geom",
    "failed_instances",
    "avg_sdp_iters",
    "avg_lmo_calls",
    "avg_nodes",
    "avg_cuts",
])

# All metrics that can be shown; order and short header for LaTeX; 5th elem = :max/:min to bold best per column (or nothing)
const METRIC_CONFIG = [
    ("pct_solved", "% sol.", "1.1f", :pct, :max),
    ("time_geom_mean", "time (s)", "0.1f", :time, :min),
    ("time_std_wrt_geom", "time std", "0.2f", :numeric, nothing),
    ("rel_gap_geom_mean_unsolved", "rel. gap", "0.2e", :scientific, :min),
    ("failed_instances", "failed", "d", :int, nothing),
    ("avg_lmo_calls", "LMO", "d", :int, nothing),
    ("avg_nodes", "nodes", "d", :int, nothing),
    ("avg_cuts", "cuts", "d", :int, nothing),
    ("avg_sdp_iters", "SDP it.", "d", :int, nothing),
]

function parse_hide(hide_input::Union{String,AbstractVector{<:AbstractString},Nothing})
    hide_input === nothing && return String[]
    if hide_input isa AbstractVector
        # Join and split by comma or whitespace so "a" "b,c" -> ["a", "b", "c"]
        combined = join(hide_input, " ")
    else
        combined = hide_input
    end
    isempty(strip(combined)) && return String[]
    # Split on comma or whitespace, strip, drop empty
    return [strip(s) for s in split(combined, r"[\s,]+") if !isempty(strip(s))]
end

function format_cell(val, fmt_type)
    if val === missing || (val isa Number && !isfinite(val))
        return "---"
    end
    if val isa Integer
        return string(val)
    end
    if val isa AbstractFloat
        if fmt_type == :int
            return string(round(Int, val))
        end
        if fmt_type == :pct
            return @sprintf "%.1f" val
        end
        if fmt_type == :time
            return @sprintf "%.1f" val
        end
        if fmt_type == :scientific
            return @sprintf "%.2e" val
        end
        return @sprintf "%.2f" val
    end
    return string(val)
end

function pivot_aggregated(df::DataFrame, row_col::Symbol, hide::Vector{String}, solvers::Vector{String})
    metrics = [m[1] for m in METRIC_CONFIG if m[1] in names(df) && !(m[1] in hide)]
    raw_vals = unique(df[!, row_col])
    row_vals = if row_col == :N_construction
        order = ["rank_deficient", "one", "log"]
        sort(collect(raw_vals); by=x -> (idx = findfirst(==(string(x)), order); idx === nothing ? 4 : idx))
    else
        # Put the special overall column (`-1`) last when present.
        sort(collect(raw_vals); by=x -> (x isa Number && x == -1) ? typemax(Int) : (x isa Number ? Int(x) : 0))
    end
    rows = []
    for rv in row_vals
        row_data = Any[rv]
        for solver in solvers
            sub = df[(df[!, row_col] .== rv) .& (df.solver .== solver), :]
            for (key, _, _, _, _) in METRIC_CONFIG
                key in metrics || continue
                if nrow(sub) >= 1 && key in names(sub)
                    push!(row_data, sub[1, key])
                else
                    push!(row_data, missing)
                end
            end
        end
        push!(rows, row_data)
    end
    return row_vals, metrics, rows
end

function tex_escape(s)
    s = string(s)
    s = replace(s, "&" => "\\&")
    s = replace(s, "_" => "\\_")
    s = replace(s, "%" => "\\%")
    return s
end

function write_tex_table(io, row_vals, metrics, rows, row_col::Symbol, title::String; solvers::Vector{String}, solver_labels::Dict=SOLVER_LABELS)
    n_metrics = length(metrics)
    n_solvers = length(solvers)
    n_col_vals = length(row_vals)
    # Precompute best value per (metric, column) for highlighting; best_dir from config (:max or :min)
    best_per_col = Matrix{Union{Missing,Float64}}(undef, n_metrics, n_col_vals)
    best_per_col .= missing
    for (m_idx, key) in enumerate(metrics)
        cfg = findfirst(x -> x[1] == key, METRIC_CONFIG)
        best_dir = cfg === nothing ? nothing : METRIC_CONFIG[cfg][5]
        best_dir === nothing && continue
        for dim_idx in 1:n_col_vals
            vals = [rows[dim_idx][2 + (s - 1) * n_metrics + m_idx - 1] for s in 1:n_solvers]
            numeric = [v for v in vals if v isa Number && isfinite(v)]
            isempty(numeric) && continue
            best_per_col[m_idx, dim_idx] = best_dir == :max ? maximum(numeric) : minimum(numeric)
        end
    end
    # Column-level flags used to suppress specific bolding in edge cases:
    # 1) If all setups time out (pct_solved == 0 for all), do not bold `% sol.` / `time (s)`.
    # 2) If any setup solved all (pct_solved == 100), do not bold `rel. gap`.
    all_timeout_col = fill(false, n_col_vals)
    any_full_solved_col = fill(false, n_col_vals)
    pct_idx = findfirst(==("pct_solved"), metrics)
    if pct_idx !== nothing
        for dim_idx in 1:n_col_vals
            vals = [rows[dim_idx][2 + (s - 1) * n_metrics + pct_idx - 1] for s in 1:n_solvers]
            numeric = [v for v in vals if v isa Number && isfinite(v)]
            if !isempty(numeric)
                all_timeout_col[dim_idx] = all(v -> isapprox(v, 0.0; atol=1e-9), numeric)
                any_full_solved_col[dim_idx] = any(v -> isapprox(v, 100.0; atol=1e-9), numeric)
            end
        end
    end
    # Transposed: columns = row_vals (dimension or N_construction), rows = solver blocks with one row per metric
    col_spec = "ll" * repeat("r", n_col_vals)
    println(io, "\\begin{tabular}{", col_spec, "}")
    println(io, "\\toprule")
    # Header: Solver, Metric, then one column per dimension (or N_construction)
    header_vals = [((v isa Number && v == -1) ? "all" : string(v)) for v in row_vals]
    line1 = " Solver & Metric & " * join(tex_escape.(header_vals), " & ")
    println(io, line1, " \\\\")
    println(io, "\\midrule")
    # Data: for each solver, multirow block with one row per metric (first col = solver multirow, second = metric name)
    for (solver_idx, solver) in enumerate(solvers)
        data_idx_base = 2 + (solver_idx - 1) * n_metrics
        row_in_block = 0
        for (key, label, _, fmt, best_dir) in METRIC_CONFIG
            key in metrics || continue
            row_in_block += 1
            metric_local_idx = findfirst(==(key), metrics)
            col_cells = String[]
            for (dim_idx, _) in enumerate(row_vals)
                val = rows[dim_idx][data_idx_base + metric_local_idx - 1]
                cell = format_cell(val, fmt)
                best = best_per_col[metric_local_idx, dim_idx]
                if best_dir !== nothing && best !== missing && val isa Number && isfinite(val)
                    allow_bold = true
                    if key in ("pct_solved", "time_geom_mean") && all_timeout_col[dim_idx]
                        allow_bold = false
                    elseif key == "rel_gap_geom_mean_unsolved" && any_full_solved_col[dim_idx]
                        allow_bold = false
                    end
                    if allow_bold && ((best_dir == :max && isapprox(val, best, rtol=1e-9)) || (best_dir == :min && isapprox(val, best, rtol=1e-9)))
                        cell = "\\textbf{$cell}"
                    end
                end
                push!(col_cells, cell)
            end
            first_cell = if row_in_block == 1
                sl = get(solver_labels, solver, solver)
                "\\multirow{$(n_metrics)}{*}{$(sl)}"
            else
                " "
            end
            line = first_cell * " & " * tex_escape(label) * " & " * join(col_cells, " & ")
            println(io, line, " \\\\")
        end
        # Visual separator between solver blocks (not after the last one)
        if solver_idx < n_solvers
            println(io, "\\midrule")
        end
    end
    println(io, "\\bottomrule")
    println(io, "\\end{tabular}")
end

function main(; data_type="both", hide=nothing, out_dir=nothing, smoothing=false, agc=false, acst_trees=false, all_boscia=false)
    hide_list = String.(unique(vcat(collect(DEFAULT_HIDDEN_METRICS), parse_hide(hide))))
    out = something(out_dir, TEX_OUT_DIR)
    mkpath(out)
    solver_order = all_boscia ? SOLVERS_WITH_EXCLUSIONS : SOLVERS_DEFAULT
    if agc
        # AGC: two setups, by dimension only
        setups = [
            ("correlated_connected", "AGC correlated connected"),
            ("independent_disconnected", "AGC independent disconnected"),
        ]
        for (tag, title) in setups
            path = joinpath(AGG_DIR, "agc_$(tag)_by_dimension.csv")
            isfile(path) || continue
            df = CSV.read(path, DataFrame)
            solver_set = Set(String.(unique(df.solver)))
            solvers = [s for s in solver_order if s in solver_set]
            row_vals, metrics, rows = pivot_aggregated(df, :dimension, hide_list, solvers)
            tex_path = joinpath(out, "agc_$(tag)_by_dimension.tex")
            open(tex_path, "w") do io
                write_tex_table(io, row_vals, metrics, rows, :dimension, title; solvers=solvers, solver_labels=SOLVER_LABELS)
            end
            println("Wrote ", tex_path)

            cmp_path = joinpath(AGG_DIR, "agc_$(tag)_reduced_vs_baseline_by_dimension.csv")
            if isfile(cmp_path)
                cmp_df = CSV.read(cmp_path, DataFrame)
                cmp_solver_set = Set(String.(unique(cmp_df.solver)))
                cmp_solvers = [s for s in SOLVERS_REDUCED_VS_BASELINE if s in cmp_solver_set]
                if !isempty(cmp_solvers)
                    row_vals_cmp, metrics_cmp, rows_cmp = pivot_aggregated(cmp_df, :dimension, hide_list, cmp_solvers)
                    cmp_tex_path = joinpath(out, "agc_$(tag)_reduced_vs_baseline_by_dimension.tex")
                    open(cmp_tex_path, "w") do io
                        write_tex_table(io, row_vals_cmp, metrics_cmp, rows_cmp, :dimension, "AGC reduced spectrum vs baseline ($(tag))"; solvers=cmp_solvers, solver_labels=SOLVER_LABELS)
                    end
                    println("Wrote ", cmp_tex_path)
                end
            end
        end
    elseif acst_trees
        order = all_boscia ? SOLVERS_TEX_SPANNING_TREE_ALL : SOLVERS_TEX_SPANNING_TREE
        path = joinpath(AGG_DIR, "spanning_tree_independent_by_dimension.csv")
        if isfile(path)
            df = CSV.read(path, DataFrame)
            df = df[df.dimension .< 780, :]
            solver_set = Set(String.(unique(df.solver)))
            solvers = [s for s in order if s in solver_set]
            row_vals, metrics, rows = pivot_aggregated(df, :dimension, hide_list, solvers)
            tex_path = joinpath(out, "spanning_tree_independent_by_dimension.tex")
            open(tex_path, "w") do io
                write_tex_table(io, row_vals, metrics, rows, :dimension, "Spanning tree — Boscia ACST and ACSTS (independent)"; solvers=solvers, solver_labels=SOLVER_LABELS)
            end
            println("Wrote ", tex_path)
        else
            println("Skip spanning-tree TeX (Boscia formulations): $path not found")
        end
        path_paj = joinpath(AGG_DIR, "spanning_tree_acst_rank_pruning_vs_scipsdp_independent_by_dimension.csv")
        if isfile(path_paj)
            df_p = CSV.read(path_paj, DataFrame)
            df_p = df_p[df_p.dimension .< 780, :]
            solver_set_p = Set(String.(unique(df_p.solver)))
            solvers_p = [s for s in SOLVERS_TEX_ACST_RANK_VS_SCIPSDP if s in solver_set_p]
            row_vals_p, metrics_p, rows_p = pivot_aggregated(df_p, :dimension, hide_list, solvers_p)
            tex_paj = joinpath(out, "spanning_tree_acst_rank_pruning_vs_scipsdp_independent_by_dimension.tex")
            open(tex_paj, "w") do io
                write_tex_table(io, row_vals_p, metrics_p, rows_p, :dimension, "Spanning tree — ACST Boscia vs SCIPSDP (independent)"; solvers=solvers_p, solver_labels=SOLVER_LABELS)
            end
            println("Wrote ", tex_paj)
        else
            println("Skip ACST vs SCIPSDP TeX: $path_paj not found")
        end
        path_cmp = joinpath(AGG_DIR, "spanning_tree_acst_reduced_vs_baseline_independent_by_dimension.csv")
        if isfile(path_cmp)
            df_c = CSV.read(path_cmp, DataFrame)
            df_c = df_c[df_c.dimension .< 780, :]
            solver_set_c = Set(String.(unique(df_c.solver)))
            solvers_c = [s for s in SOLVERS_TEX_ACST_REDUCED_VS_BASELINE if s in solver_set_c]
            row_vals_c, metrics_c, rows_c = pivot_aggregated(df_c, :dimension, hide_list, solvers_c)
            tex_cmp = joinpath(out, "spanning_tree_acst_reduced_vs_baseline_independent_by_dimension.tex")
            open(tex_cmp, "w") do io
                write_tex_table(io, row_vals_c, metrics_c, rows_c, :dimension, "Spanning tree — ACST reduced spectrum vs baseline (independent)"; solvers=solvers_c, solver_labels=SOLVER_LABELS)
            end
            println("Wrote ", tex_cmp)
        else
            println("Skip ACST reduced-vs-baseline TeX: $path_cmp not found")
        end
    else
        prefix = smoothing ? "smoothing_" : ""
        solver_labels = smoothing ? SMOOTHING_LABELS : SOLVER_LABELS
        data_types = data_type == "both" ? ["independent", "correlated"] : [data_type]
        for dtype in data_types
            for (row_col, suffix, title_suffix) in [
                (:dimension, "dimension", "by dimension"),
                (:N_construction, "N_construction", "by N construction"),
            ]
                path = joinpath(AGG_DIR, "$(prefix)$(dtype)_by_$(suffix).csv")
                isfile(path) || continue
                df = CSV.read(path, DataFrame)
                # Use only solvers that appear in the data (preserve configured order)
                solver_set = Set(String.(unique(df.solver)))
                solvers = smoothing ? SMOOTHING_REGIMES : [s for s in solver_order if s in solver_set]
                row_vals, metrics, rows = pivot_aggregated(df, row_col, hide_list, solvers)
                tex_path = joinpath(out, "$(prefix)$(dtype)_by_$(suffix).tex")
                open(tex_path, "w") do io
                    write_tex_table(io, row_vals, metrics, rows, row_col, "$(title_suffix) ($(dtype))"; solvers=solvers, solver_labels=solver_labels)
                end
                println("Wrote ", tex_path)
            end

            # Additional diagnostic table: reduced_spectrum ("Boscia") vs baseline Boscia
            cmp_path = joinpath(AGG_DIR, "$(dtype)_reduced_vs_baseline_by_dimension.csv")
            if isfile(cmp_path)
                cmp_df = CSV.read(cmp_path, DataFrame)
                cmp_solver_set = Set(String.(unique(cmp_df.solver)))
                cmp_solvers = [s for s in SOLVERS_REDUCED_VS_BASELINE if s in cmp_solver_set]
                if !isempty(cmp_solvers)
                    row_vals_cmp, metrics_cmp, rows_cmp = pivot_aggregated(cmp_df, :dimension, hide_list, cmp_solvers)
                    cmp_tex_path = joinpath(out, "$(dtype)_reduced_vs_baseline_by_dimension.tex")
                    open(cmp_tex_path, "w") do io
                        write_tex_table(io, row_vals_cmp, metrics_cmp, rows_cmp, :dimension, "Reduced spectrum vs baseline ($(dtype))"; solvers=cmp_solvers, solver_labels=SOLVER_LABELS)
                    end
                    println("Wrote ", cmp_tex_path)
                end
            end
        end
    end
    println("\n--- Required LaTeX preamble ---")
    println("""
% In your .tex document preamble:
\\usepackage{booktabs}
\\usepackage{siunitx}
\\usepackage{multirow}

% Optional: for narrow tables or long captions
% \\usepackage{caption}
% \\captionsetup{font=small}

% Include a generated table in the document body, e.g.:
% \\input{csv/aggregated/independent_by_dimension}
% or use the path relative to your .tex file.
""")
end

# CLI
if abspath(PROGRAM_FILE) == @__FILE__
    function parse_args(args)
        hide_arg = nothing
        data_type_arg = "both"
        out_dir_arg = nothing
        smoothing_arg = false
        agc_arg = false
        acst_trees_arg = false
        acst_arg = false
        acsts_arg = false
        all_boscia_arg = false
        idx = 1
        while idx <= length(args)
            if args[idx] == "--hide"
                idx += 1
                hide_list = String[]
                while idx <= length(args) && !startswith(args[idx], "--")
                    push!(hide_list, args[idx])
                    idx += 1
                end
                hide_arg = isempty(hide_list) ? nothing : hide_list
            elseif args[idx] == "--data" && idx + 1 <= length(args)
                data_type_arg = args[idx + 1]
                idx += 2
            elseif args[idx] == "--out" && idx + 1 <= length(args)
                out_dir_arg = args[idx + 1]
                idx += 2
            elseif args[idx] == "--smoothing"
                smoothing_arg = true
                idx += 1
            elseif args[idx] == "--agc"
                agc_arg = true
                idx += 1
            elseif args[idx] == "--acst-trees"
                acst_trees_arg = true
                idx += 1
            elseif args[idx] == "--acst"
                acst_arg = true
                idx += 1
            elseif args[idx] == "--acsts"
                acsts_arg = true
                idx += 1
            elseif args[idx] == "--all-boscia"
                all_boscia_arg = true
                idx += 1
            else
                idx += 1
            end
        end
        spanning = acst_trees_arg || acst_arg || acsts_arg
        return (; data_type=data_type_arg, hide=hide_arg, out_dir=out_dir_arg, smoothing=smoothing_arg, agc=agc_arg, acst_trees=spanning, all_boscia=all_boscia_arg)
    end
    opts = parse_args(ARGS)
    main(; opts...)
end
