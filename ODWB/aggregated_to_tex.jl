#!/usr/bin/env julia
#=
Generate LaTeX tables from aggregated CSVs (by_dimension and by_N_construction).
- Each solver (Boscia, SCIPSDP_oa, SCIPSDP_bnb) is a multi-column block.
- Rows = dimension or N_construction.
- Option to hide columns via --hide (comma-separated list).

Hideable column names: pct_solved, time_geom_mean, time_std_wrt_geom,
  rel_gap_geom_mean_unsolved, failed_instances, avg_lmo_calls, avg_nodes,
  avg_cuts, avg_sdp_iters.

Usage:
  julia aggregated_to_tex.jl [--hide COL1 COL2 ...] [--data independent|correlated|both] [--out DIR]
  julia aggregated_to_tex.jl --hide avg_nodes time_std_wrt_geom
  julia aggregated_to_tex.jl --hide avg_lmo_calls avg_nodes avg_cuts avg_sdp_iters

All arguments after --hide (until the next --) are treated as column names to hide; commas are optional.

Output: writes .tex files into the paper repo (or --out DIR) and prints required preamble.
=#

using CSV, DataFrames, Printf

const AGG_DIR = joinpath(@__DIR__, "csv", "aggregated")
const TEX_OUT_DIR = "/Users/deborah/Documents/research_projects/Smoothing-in-Boscia/paper"
const SOLVERS = ["Boscia", "SCIPSDP_oa", "SCIPSDP_bnb"]
const SOLVER_LABELS = Dict("Boscia" => "Boscia", "SCIPSDP_oa" => "SCIPSDP (OA)", "SCIPSDP_bnb" => "SCIPSDP (B\\&B)")

# All metrics that can be shown; order and short header for LaTeX; 5th elem = :max/:min to bold best per column (or nothing)
const METRIC_CONFIG = [
    ("pct_solved", "% sol.", "1.1f", :pct, :max),
    ("time_geom_mean", "time (s)", "0.1f", :time, :min),
    ("time_std_wrt_geom", "time std", "0.2f", :numeric, nothing),
    ("rel_gap_geom_mean_unsolved", "rel. gap", "0.2e", :scientific, nothing),
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

function pivot_aggregated(df::DataFrame, row_col::Symbol, hide::Vector{String})
    metrics = [m[1] for m in METRIC_CONFIG if m[1] in names(df) && !(m[1] in hide)]
    raw_vals = unique(df[!, row_col])
    row_vals = if row_col == :N_construction
        order = ["rank_deficient", "one", "log"]
        sort(collect(raw_vals); by=x -> (idx = findfirst(==(string(x)), order); idx === nothing ? 4 : idx))
    else
        sort(collect(raw_vals))
    end
    rows = []
    for rv in row_vals
        row_data = Any[rv]
        for solver in SOLVERS
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

function write_tex_table(io, row_vals, metrics, rows, row_col::Symbol, title::String)
    n_metrics = length(metrics)
    n_solvers = length(SOLVERS)
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
    # Transposed: columns = row_vals (dimension or N_construction), rows = solver blocks with one row per metric
    col_spec = "ll" * repeat("r", n_col_vals)
    println(io, "\\begin{tabular}{", col_spec, "}")
    println(io, "\\toprule")
    # Header: Solver, Metric, then one column per dimension (or N_construction)
    line1 = " Solver & Metric & " * join(tex_escape.(string.(row_vals)), " & ")
    println(io, line1, " \\\\")
    println(io, "\\midrule")
    # Data: for each solver, multirow block with one row per metric (first col = solver multirow, second = metric name)
    for (solver_idx, solver) in enumerate(SOLVERS)
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
                    if (best_dir == :max && isapprox(val, best, rtol=1e-9)) || (best_dir == :min && isapprox(val, best, rtol=1e-9))
                        cell = "\\textbf{$cell}"
                    end
                end
                push!(col_cells, cell)
            end
            first_cell = if row_in_block == 1
                "\\multirow{$(n_metrics)}{*}{$(SOLVER_LABELS[solver])}"
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

function main(; data_type="both", hide=nothing, out_dir=nothing)
    hide_list = String.(parse_hide(hide))
    out = something(out_dir, TEX_OUT_DIR)
    mkpath(out)
    data_types = data_type == "both" ? ["independent", "correlated"] : [data_type]
    for dtype in data_types
        for (row_col, suffix, title_suffix) in [
            (:dimension, "dimension", "by dimension"),
            (:N_construction, "N_construction", "by N construction"),
        ]
            path = joinpath(AGG_DIR, "$(dtype)_by_$(suffix).csv")
            isfile(path) || continue
            df = CSV.read(path, DataFrame)
            row_vals, metrics, rows = pivot_aggregated(df, row_col, hide_list)
            tex_path = joinpath(out, "$(dtype)_by_$(suffix).tex")
            open(tex_path, "w") do io
                write_tex_table(io, row_vals, metrics, rows, row_col, "$(title_suffix) ($(dtype))")
            end
            println("Wrote ", tex_path)
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
        idx = 1
        while idx <= length(args)
            if args[idx] == "--hide"
                # Consume all following args until next -- (or end) as the hide list
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
            else
                idx += 1
            end
        end
        return (; data_type=data_type_arg, hide=hide_arg, out_dir=out_dir_arg)
    end
    opts = parse_args(ARGS)
    main(; opts...)
end
