#!/usr/bin/env julia
#=
Build one publication table comparing smoothing/truncated-gradient ablations.

Metrics:
- percentage solved;
- arithmetic mean time over the full official instance grid (missing/timeouts = 3600s);
- arithmetic mean relative dual gap over unsolved rows with a finite reported gap.

An OPTIMAL claim that is more than 5% worse than another setting's feasible
objective on the same instance is treated as unsolved and timed out.
=#

using CSV
using DataFrames
using Printf
using Statistics

const BOSCIA_DIR = joinpath(@__DIR__, "csv", "Boscia")
const TIME_LIMIT = 3600.0
const REL_TOL = 0.05
const OPTIMAL_TERMINATIONS = Set(["OPTIMAL", "GAPLIMIT", "OPTIMALITY_PROVED"])

const SETTINGS = [
    ("baseline", "Baseline"),
    ("scaled_mu", "Adaptive \$\\mu\$"),
    ("reduced_spectrum_2", "Half spectrum"),
    ("reduced_spectrum_3", "Third spectrum"),
    ("reduced_spectrum_2_scaled_mu", "Half spectrum + adaptive \$\\mu\$"),
    ("reduced_spectrum_3_scaled_mu", "Third spectrum + adaptive \$\\mu\$"),
]

const CASES = [
    (id="E_corr", label="EOD corr.", criterion="E", suffix="correlated__", sense=:min,
     keys=[(m, floor(Int, sqrt(m)), N, seed)
           for m in (50, 80, 100, 120, 150)
           for N in (floor(Int, 1.5 * floor(Int, sqrt(m))),
                     floor(Int, 1.5 * floor(Int, sqrt(m)) * log(floor(Int, sqrt(m)))))
           for seed in 1:5]),
    (id="E_ind", label="EOD ind.", criterion="E", suffix="independent__", sense=:min,
     keys=[(m, floor(Int, sqrt(m)), N, seed)
           for m in (50, 80, 100, 120, 150)
           for N in (floor(Int, 1.5 * floor(Int, sqrt(m))),
                     floor(Int, 1.5 * floor(Int, sqrt(m)) * log(floor(Int, sqrt(m)))))
           for seed in 1:5]),
    (id="AGC_corr", label="AGC corr.", criterion="AGC", suffix="correlated_connected", sense=:min,
     keys=[(m, floor(Int, m / 3), floor(Int, m / 2), seed)
           for m in (80, 100, 150, 200) for seed in 1:5]),
    (id="AGC_ind", label="AGC ind.", criterion="AGC", suffix="independent_disconnected", sense=:min,
     keys=[(m, floor(Int, m / 3), floor(Int, m / 2), seed)
           for m in (80, 100, 150, 200) for seed in 1:5]),
    # Boscia records the negated algebraic-connectivity objective, hence :min.
    (id="ACST", label="ACST", criterion="ACST", suffix="independent__", sense=:min,
     keys=[(n * (n - 1) ÷ 2, n, n - 1, seed)
           for n in (10, 12, 15, 25, 40, 60, 100) for seed in 1:3]),
    (id="ACSTS", label="ACSTS", criterion="ACSTS", suffix="independent__", sense=:min,
     keys=[(n * (n - 1) ÷ 2, n, n - 1, seed)
           for n in (10, 12, 15, 25, 40, 60, 100) for seed in 1:3]),
]

Base.@kwdef mutable struct RunRecord
    time::Float64 = TIME_LIMIT
    termination::String = "ERROR"
    objective::Float64 = Inf
    rel_gap::Float64 = Inf
    present::Bool = false
end

is_solved(r::RunRecord) =
    r.termination in OPTIMAL_TERMINATIONS && isfinite(r.time) && r.time < TIME_LIMIT

parse_float(x, default=NaN) = begin
    (x === missing || x === nothing) && return default
    v = tryparse(Float64, string(x))
    isnothing(v) ? default : v
end

function variant_regex(setting::String, criterion::String, suffix::String)
    prefix = setting == "baseline" ? "boscia__" : "boscia_$(setting)_"
    separator = endswith(suffix, "_") ? "" : "_"
    optional_scaling = raw"(?:_scaled_[^_]+_)?"
    return Regex("^" * prefix * criterion * optional_scaling * "_optimality_" *
                 suffix * separator * raw"(\d+)_(\d+)_(\d+)_(\d+)\.csv$")
end

"""
Load all matching singles. If old scaled-tag and corrected untagged filenames
exist for the same key, prefer the corrected untagged filename.
"""
function load_variant(setting::String, case)
    rx = variant_regex(setting, case.criterion, case.suffix)
    matches = Dict{NTuple{4,Int},Tuple{String,Bool}}()
    for fname in readdir(BOSCIA_DIR)
        m = match(rx, fname)
        isnothing(m) && continue
        key = Tuple(parse.(Int, m.captures))::NTuple{4,Int}
        key in case.keys || continue
        has_scaled_tag = occursin(r"_scaled_[^_]+__optimality", fname)
        old = get(matches, key, nothing)
        if old === nothing || (old[2] && !has_scaled_tag)
            matches[key] = (fname, has_scaled_tag)
        end
    end

    out = Dict{NTuple{4,Int},RunRecord}()
    for key in case.keys
        candidate = get(matches, key, nothing)
        if candidate === nothing
            out[key] = RunRecord()
            continue
        end
        path = joinpath(BOSCIA_DIR, candidate[1])
        df = CSV.read(path, DataFrame; delim=';', silencewarnings=true)
        if nrow(df) == 0
            out[key] = RunRecord()
            continue
        end
        row = df[end, :]
        out[key] = RunRecord(
            time=min(parse_float(row.time, TIME_LIMIT), TIME_LIMIT),
            termination=string(row.termination),
            # `solution` is in the original problem scale. `scaled_solution`
            # is not comparable across scaled-μ and unscaled runs.
            objective=parse_float(row.solution, Inf),
            rel_gap=hasproperty(row, :rel_dual_gap) ?
                parse_float(row.rel_dual_gap, Inf) : Inf,
            present=true,
        )
    end
    return out
end

function apply_cross_setting_check!(runs_by_setting, case)
    for key in case.keys
        objective_values = [runs[key].objective for runs in Base.values(runs_by_setting)
                            if isfinite(runs[key].objective)]
        isempty(objective_values) && continue
        best = case.sense == :min ? minimum(objective_values) : maximum(objective_values)
        for runs in values(runs_by_setting)
            r = runs[key]
            r.termination in ("OPTIMAL", "OPTIMALITY_PROVED") || continue
            isfinite(r.objective) || continue
            tol = max(1e-5, REL_TOL * max(abs(best), abs(r.objective), 1e-12))
            inconsistent = case.sense == :min ?
                r.objective > best + tol :
                r.objective < best - tol
            inconsistent && (r.time = TIME_LIMIT)
        end
    end
end

function summarize_setting(runs, case)
    records = [runs[key] for key in case.keys]
    solved = is_solved.(records)
    pct_solved = 100 * count(solved) / length(records)
    mean_time = mean(r.time for r in records)
    finite_unsolved_gaps = [r.rel_gap for (r, s) in zip(records, solved)
                            if !s && isfinite(r.rel_gap) && r.rel_gap >= 0]
    mean_gap = isempty(finite_unsolved_gaps) ? missing : mean(finite_unsolved_gaps)
    n_present = count(r.present for r in records)
    return (; pct_solved, mean_time, mean_gap, n_present, n_expected=length(records))
end

format_pct(x) = @sprintf("%.1f", x)
format_time(x) = @sprintf("%.1f", x)
format_gap(x) = ismissing(x) ? "---" : @sprintf("%.2e", x)

function maybe_bold(text, value, best; direction)
    (ismissing(value) || ismissing(best)) && return text
    equal = isapprox(value, best; rtol=1e-9, atol=1e-12)
    return equal ? "\\textbf{$text}" : text
end

function write_tex(path, summaries)
    case_labels = [c.label for c in CASES]
    open(path, "w") do io
        println(io, "\\begin{tabular}{ll", repeat("r", length(CASES)), "}")
        println(io, "\\toprule")
        println(io, " Setting & Metric & ", join(case_labels, " & "), " \\\\")
        println(io, "\\midrule")

        best_pct = Dict(c.id => maximum(summaries[(s, c.id)].pct_solved for (s, _) in SETTINGS) for c in CASES)
        best_time = Dict(c.id => minimum(summaries[(s, c.id)].mean_time for (s, _) in SETTINGS) for c in CASES)
        best_gap = Dict{String,Any}()
        for c in CASES
            vals = [summaries[(s, c.id)].mean_gap for (s, _) in SETTINGS]
            finite_vals = [v for v in vals if !ismissing(v)]
            best_gap[c.id] = isempty(finite_vals) ? missing : minimum(finite_vals)
        end

        for (setting_idx, (setting, label)) in enumerate(SETTINGS)
            pct_cells = String[]
            time_cells = String[]
            gap_cells = String[]
            for c in CASES
                s = summaries[(setting, c.id)]
                push!(pct_cells, maybe_bold(format_pct(s.pct_solved), s.pct_solved, best_pct[c.id]; direction=:max))
                push!(time_cells, maybe_bold(format_time(s.mean_time), s.mean_time, best_time[c.id]; direction=:min))
                push!(gap_cells, maybe_bold(format_gap(s.mean_gap), s.mean_gap, best_gap[c.id]; direction=:min))
            end
            println(io, "\\multirow{3}{*}{$label} & \\% sol. & ", join(pct_cells, " & "), " \\\\")
            println(io, " & time (s) & ", join(time_cells, " & "), " \\\\")
            println(io, " & rel. gap & ", join(gap_cells, " & "), " \\\\")
            setting_idx < length(SETTINGS) && println(io, "\\midrule")
        end
        println(io, "\\bottomrule")
        println(io, "\\end{tabular}")
    end
end

function main()
    summaries = Dict{Tuple{String,String},Any}()
    csv_rows = NamedTuple[]
    for case in CASES
        runs_by_setting = Dict(setting => load_variant(setting, case)
                               for (setting, _) in SETTINGS)
        apply_cross_setting_check!(runs_by_setting, case)
        for (setting, label) in SETTINGS
            summary = summarize_setting(runs_by_setting[setting], case)
            summaries[(setting, case.id)] = summary
            push!(csv_rows, (; setting, setting_label=label, case=case.id,
                             case_label=case.label, summary...))
            println("$(case.label), $label: $(summary.n_present)/$(summary.n_expected) present, " *
                    "$(round(summary.pct_solved; digits=1))% solved")
        end
    end

    csv_path = joinpath(@__DIR__, "csv", "aggregated", "smoothing_ablation_summary.csv")
    CSV.write(csv_path, DataFrame(csv_rows))
    tex_path = "/Users/deborah/Documents/research_projects/Smoothing-in-Boscia/paper/smoothing_ablation_summary.tex"
    write_tex(tex_path, summaries)
    println("Wrote $csv_path")
    println("Wrote $tex_path")
end

main()
