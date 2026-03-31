#!/usr/bin/env julia
#
# Plot relative gap over time and iteration for E-optimality Boscia runs
# with different mu settings (large constant, small constant, decay 0.9, decay 0.7).
# Uses only instances (m, n, N, seed) where all four settings exist for the given data type.
#
# Usage:
#   julia plot_mu_testing_rel_gap.jl [data_type] [instance_spec]
#   data_type: "independent" | "correlated" | "both" (default: both)
#   instance_spec: optional, e.g. "50_7_20_1" to plot only that instance; else all common instances.
#
# Example: julia plot_mu_testing_rel_gap.jl both
#          julia plot_mu_testing_rel_gap.jl independent 50_7_20_1
#

using CSV, DataFrames, Printf
using Colors
include(joinpath(@__DIR__, "colours.jl"))

# Plotting: try PyPlot, then Plots (with GR or pyplot backend)
const PLOT_BACKEND = try
    using PyPlot
    "PyPlot"
catch
    try
        using Plots
        Plots.gr()
        "Plots"
    catch
        ""
    end
end
const CAN_PLOT = !isempty(PLOT_BACKEND)

const FULL_RUNS_DIR = joinpath(@__DIR__, "csv", "full_runs_boscia")

# Mu setting labels for plots (order: large const, small const, decay 0.9, decay 0.7)
const MU_SETTING_LABELS = [
    "large constant μ",
    "small constant μ",
    "decay 0.9",
    "decay 0.7",
]

"""Parse folder string 'start_decay_min' into (start, decay, min) and classify into setting type."""
function classify_folder(folder::AbstractString, m::Int)
    parts = split(folder, "_")
    if length(parts) < 3
        return nothing
    end
    start_val = try
        parse(Float64, parts[1])
    catch
        return nothing
    end
    decay_val = try
        parse(Float64, parts[2])
    catch
        return nothing
    end
    if decay_val ≈ 0.9
        return "decay_0.9"
    elseif decay_val ≈ 0.7
        return "decay_0.7"
    elseif decay_val ≈ 1.0
        large_start = m / 50.0
        if isapprox(start_val, large_start; rtol = 0.2)
            return "large_const"
        else
            return "small_const"
        end
    end
    return nothing
end

"""Parse filename to get folder, data_type, m, n, N, seed. Returns nothing if not E_optimality with __m_n_N_seed."""
function parse_eopt_filename(filename::AbstractString)
    if !occursin("_E_optimality_", filename) || !occursin("__", filename)
        return nothing
    end
    # Pattern: boscia_<folder>_E_optimality_<independent|correlated>__m_n_N_seed.csv
    # or boscia__E_optimality_... (empty folder)
    base = replace(filename, ".csv" => "")
    if !occursin("__", base)
        return nothing
    end
    left, right = split(base, "__"; limit = 2)
    if right === nothing || isempty(right)
        return nothing
    end
    # right = "m_n_N_seed"
    nums = split(right, "_")
    if length(nums) != 4
        return nothing
    end
    m = try
        parse(Int, nums[1])
    catch
        return nothing
    end
    n = try
        parse(Int, nums[2])
    catch
        return nothing
    end
    N = try
        parse(Int, nums[3])
    catch
        return nothing
    end
    seed = try
        parse(Int, nums[4])
    catch
        return nothing
    end
    # data type
    if occursin("_independent__", base) || endswith(left, "_independent")
        data_type = "independent"
    elseif occursin("_correlated__", base) || endswith(left, "_correlated")
        data_type = "correlated"
    else
        return nothing
    end
    # folder: between "boscia_" and "_E_optimality_"
    if !startswith(base, "boscia_") || !occursin("_E_optimality_", base)
        return nothing
    end
    idx = findfirst("_E_optimality_", base)
    folder = base[8:(first(idx) - 1)]
    if isempty(folder)
        return nothing  # skip baseline (no mu_testing folder)
    end
    return (; folder, data_type, m, n, N, seed)
end

"""Scan full_runs_boscia and return DataFrame with columns: file, folder, data_type, m, n, N, seed, setting."""
function scan_mu_testing_files()
    if !isdir(FULL_RUNS_DIR)
        @warn "Directory not found: $FULL_RUNS_DIR"
        return DataFrame()
    end
    all_files = readdir(FULL_RUNS_DIR)
    rows = []
    for f in all_files
        if !endswith(f, ".csv")
            continue
        end
        info = parse_eopt_filename(f)
        if info === nothing
            continue
        end
        setting = classify_folder(info.folder, info.m)
        if setting === nothing
            continue
        end
        push!(rows, (;
            file = f,
            folder = info.folder,
            data_type = info.data_type,
            m = info.m,
            n = info.n,
            N = info.N,
            seed = info.seed,
            setting = setting,
        ))
    end
    return DataFrame(rows)
end

"""For each data_type, find (m, n, N, seed) that have all four settings."""
function find_common_instances(df::DataFrame)
    required = Set(["large_const", "small_const", "decay_0.9", "decay_0.7"])
    by_type = Dict{String, Set{Tuple{Int,Int,Int,Int}}}()
    for dt in ["independent", "correlated"]
        sub = df[df.data_type .== dt, :]
        instances = Set{Tuple{Int,Int,Int,Int}}()
        for g in groupby(sub, [:m, :n, :N, :seed])
            if Set(g.setting) == required
                push!(instances, (g.m[1], g.n[1], g.N[1], g.seed[1]))
            end
        end
        by_type[dt] = instances
    end
    return by_type
end

"""For each data_type, find instances that have at least min_settings (for plotting when not all 4 exist)."""
function find_plotable_instances(df::DataFrame; min_settings::Int = 2)
    by_type = Dict{String, Set{Tuple{Int,Int,Int,Int}}}()
    for dt in ["independent", "correlated"]
        sub = df[df.data_type .== dt, :]
        instances = Set{Tuple{Int,Int,Int,Int}}()
        for g in groupby(sub, [:m, :n, :N, :seed])
            if length(unique(g.setting)) >= min_settings
                push!(instances, (g.m[1], g.n[1], g.N[1], g.seed[1]))
            end
        end
        by_type[dt] = instances
    end
    return by_type
end

"""Load progress CSV and compute relative gap. Returns (time, iteration, rel_gap) vectors.

Relative gap definition used here:
    rel_gap = abs(upperBound - lowerBound) / abs(min(upperBound, lowerBound))

The `abs` on the denominator ensures the gap is nonnegative (so it can be shown on a log scale),
and matches the intent of scaling by the smaller (more conservative) bound magnitude.
"""
function load_rel_gap(csv_path::String)
    df = CSV.read(csv_path, DataFrame)
    if !("lowerBound" in names(df)) || !("upperBound" in names(df))
        return Float64[], Int[], Float64[]
    end
    lb = df.lowerBound
    ub = df.upperBound
    time_raw = "time" in names(df) ? df.time : collect(1:nrow(df))
    # In our full-run CSVs, time is typically recorded in milliseconds.
    # Heuristic: if the maximum exceeds 10k, treat it as ms and convert to seconds.
    time = (maximum(time_raw) > 10_000) ? (time_raw ./ 1000.0) : Float64.(time_raw)
    n = length(lb)
    rel_gap = Float64[]
    for i in 1:n
        abs_gap = abs(ub[i] - lb[i])
        denom = abs(min(ub[i], lb[i]))
        denom = max(denom, 1e-10)
        push!(rel_gap, abs_gap / denom)
    end
    iter = 1:n
    return time, iter, rel_gap
end

"""Plot relative gap vs time and vs iteration for one instance and one data type."""
function plot_instance_rel_gap(
    files_by_setting::Dict{String, String},
    data_type::String,
    m::Int, n::Int, N::Int, seed::Int;
    save_plots::Bool = true,
    out_dir::String = joinpath(@__DIR__, "plots"),
)
    if !CAN_PLOT
        @warn "No plotting backend (PyPlot/Plots) available; skipping plot generation."
        return
    end
    # Turn (r,g,b) tuples from colours.jl into something each backend accepts.
    rgb_tuple(t) = (float(t[1]), float(t[2]), float(t[3]))
    rgb_color(t) = RGB(float(t[1]), float(t[2]), float(t[3]))
    setting_order = ["large_const", "small_const", "decay_0.9", "decay_0.7"]
    # Color-blind friendly colors (from colours.jl); do not reuse colors.
    colors = [cb_blue, cb_purple, cb_rose, cb_blue_green]

    if PLOT_BACKEND == "PyPlot"
        fig, axes = PyPlot.subplots(1, 2, figsize = (12, 5))
        for (idx, key) in enumerate(setting_order)
            path = get(files_by_setting, key, nothing)
            if path === nothing || !isfile(path)
                continue
            end
            time, iter, rel_gap = load_rel_gap(path)
            if isempty(rel_gap)
                continue
            end
            label = MU_SETTING_LABELS[idx]
            axes[1].plot(time, rel_gap, color = rgb_tuple(colors[idx]), label = label, alpha = 0.85)
            axes[2].plot(iter, rel_gap, color = rgb_tuple(colors[idx]), label = label, alpha = 0.85)
        end
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel("Relative gap")
        axes[1].set_yscale("log")
        axes[1].grid(true, alpha = 0.3)
        axes[2].set_xlabel("Iteration")
        axes[2].set_ylabel("Relative gap")
        axes[2].set_yscale("log")
        axes[2].legend()
        axes[2].grid(true, alpha = 0.3)
        PyPlot.tight_layout(rect = (0.02, 0.03, 0.98, 0.98))
        if save_plots
            mkpath(out_dir)
            filepath = joinpath(out_dir, "eopt_mu_relgap_$(data_type)_$(m)_$(n)_$(N)_$(seed).png")
            fig.savefig(filepath, dpi = 150, bbox_inches = "tight", pad_inches = 0.2)
            @info "Saved: $filepath"
        end
        PyPlot.close(fig)
    else
        # Plots.jl
        p1 = Plots.plot(; xlabel = "Time (s)", ylabel = "Relative gap", yaxis = :log, legend = false)
        p2 = Plots.plot(; xlabel = "Iteration", ylabel = "Relative gap", yaxis = :log, legend = :topright)
        for (idx, key) in enumerate(setting_order)
            path = get(files_by_setting, key, nothing)
            if path === nothing || !isfile(path)
                continue
            end
            time, iter, rel_gap = load_rel_gap(path)
            if isempty(rel_gap)
                continue
            end
            label = MU_SETTING_LABELS[idx]
            Plots.plot!(p1, time, rel_gap; label = label, color = rgb_color(colors[idx]), linewidth = 2)
            Plots.plot!(p2, iter, rel_gap; label = label, color = rgb_color(colors[idx]), linewidth = 2)
        end
        plot_combined = Plots.plot(p1, p2;
            layout = (1, 2),
            size = (1200, 540),
            left_margin = 22Plots.mm,
            bottom_margin = 18Plots.mm,
        )
        if save_plots
            mkpath(out_dir)
            filepath = joinpath(out_dir, "eopt_mu_relgap_$(data_type)_$(m)_$(n)_$(N)_$(seed).png")
            Plots.savefig(plot_combined, filepath)
            @info "Saved: $filepath"
        end
    end
    return
end

function main()
    data_type_arg = length(ARGS) >= 1 ? ARGS[1] : "both"
    instance_spec = length(ARGS) >= 2 ? ARGS[2] : nothing

    println("Scanning $FULL_RUNS_DIR for E-optimality mu_testing runs...")
    df = scan_mu_testing_files()
    if nrow(df) == 0
        println("No mu_testing E-optimality files found.")
        return
    end
    println("Found $(nrow(df)) file records.")

    by_type = find_common_instances(df)
    ind_common = by_type["independent"]
    corr_common = by_type["correlated"]
    plotable = find_plotable_instances(df; min_settings = 2)
    ind_plotable = plotable["independent"]
    corr_plotable = plotable["correlated"]
    intersection = ind_common ∩ corr_common
    println("Independent: $(length(ind_common)) instances with all 4 settings; $(length(ind_plotable)) plotable (≥2 settings).")
    println("Correlated:  $(length(corr_common)) instances with all 4 settings; $(length(corr_plotable)) plotable (≥2 settings).")
    println("Intersection (same m,n,N,seed with all 4 settings for BOTH types): $(length(intersection)) instances.")
    if length(ind_common) > 0
        println("Independent instances (m,n,N,seed): ", sort(collect(ind_common)))
    end
    if length(corr_common) > 0
        println("Correlated instances (m,n,N,seed): ", sort(collect(corr_common)))
    end
    if length(corr_plotable) > 0 && length(corr_common) == 0
        println("Correlated plotable instances (≥2 μ settings): ", sort(collect(corr_plotable)))
    end
    if length(intersection) > 0
        println("Common to both: ", sort(collect(intersection)))
    end

    if instance_spec !== nothing
        # Parse "m_n_N_seed"
        parts = split(instance_spec, "_")
        if length(parts) != 4
            println("Invalid instance_spec; use m_n_N_seed (e.g. 50_7_20_1)")
            return
        end
        m, n, N, seed = parse(Int, parts[1]), parse(Int, parts[2]), parse(Int, parts[3]), parse(Int, parts[4])
        instances_to_plot = Dict{String, Vector{Tuple{Int,Int,Int,Int}}}()
        instances_to_plot["independent"] = (m, n, N, seed) in ind_plotable ? [(m, n, N, seed)] : []
        instances_to_plot["correlated"] = (m, n, N, seed) in corr_plotable ? [(m, n, N, seed)] : []
    else
        instances_to_plot = Dict(
            "independent" => collect(ind_plotable),
            "correlated" => collect(corr_plotable),
        )
    end

    for (dt, instances) in instances_to_plot
        if data_type_arg != "both" && data_type_arg != dt
            continue
        end
        for (m, n, N, seed) in instances
            # Build path for each setting
            sub = df[(df.data_type .== dt) .& (df.m .== m) .& (df.n .== n) .& (df.N .== N) .& (df.seed .== seed), :]
            files_by_setting = Dict{String, String}()
            for row in eachrow(sub)
                files_by_setting[row.setting] = joinpath(FULL_RUNS_DIR, row.file)
            end
            if length(files_by_setting) < 2
                @warn "Instance $m $n $N $seed ($dt) has only $(length(files_by_setting)) setting(s); need ≥2 to compare; skipping."
                continue
            end
            plot_instance_rel_gap(files_by_setting, dt, m, n, N, seed)
        end
    end
    println("Done.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
