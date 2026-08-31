#!/usr/bin/env julia
#
# Plot relative gap progress for ACST/ACSTS Boscia runs from callback logs.
#
# Requested comparison:
# - ACST:  default, rank_based_pruning, eigenvalue_based_pruning
# - ACSTS: default, rank_based_pruning, eigenvalue_based_pruning, exclusion_criterion
#
# Output:
# - For n in {10,12} and seed in {1,2,3}:
#   - rel gap vs iteration (log-log; Boscia table Iter), PDF + PNG
#   - Computer Modern fonts for axis/tick/legend labels
#
# Usage:
#   julia --project=. plot_acst_acsts_relgap_progress.jl
#

using Printf
using CSV, DataFrames
using Plots
Plots.gr()
using Colors
include(joinpath(@__DIR__, "colours.jl"))
include(joinpath(@__DIR__, "..", "plot", "plot_style.jl"))

const ROOT = @__DIR__
const OUT_DIR = joinpath(ROOT, "plots")

"""
Parse a Boscia callback log and return a DataFrame with columns:
  iter::Int, time_s::Float64, rel_gap::Float64

We parse the Boscia status table lines like:
  *     1     2  ...  Gap (rel)  Time (s)  ...

We interpret `Iter` as a proxy for processed-node count (suitable for a
\"gap vs nodes\" plot even though Boscia prints \"Iter\" not \"Node\").
"""
function parse_boscia_gap_table(path::AbstractString)::DataFrame
    it = Int[]
    t = Float64[]
    g = Float64[]

    in_table = false
    iter_offset = 0
    last_iter_in_block = 0
    for line in eachline(path)
        # The header contains "Gap (rel)" and "Time (s)".
        if occursin("Gap (rel)", line) && occursin("Time (s)", line)
            in_table = true
            last_iter_in_block = 0
            continue
        end
        if !in_table
            continue
        end

        # End of a table block.
        if occursin("Solution Statistics", line) || occursin("Search Statistics", line)
            iter_offset += last_iter_in_block
            in_table = false
            continue
        end

        # Data lines usually start with '*' (incumbent) or whitespace then number.
        # Remove common ANSI control sequences (these logs sometimes contain ESC[s).
        s = replace(line, r"\x1b\[[0-9;]*[A-Za-z]" => "")
        s = strip(s)
        isempty(s) && continue
        startswith(s, "-") && continue
        # Normalize: remove leading '*' if present.
        if startswith(s, "*")
            s = strip(s[2:end])
        end
        toks = split(s)
        # Expected tokens (after optional '*'):
        # iter, open, bound, incumbent, gap_abs, gap_rel, time_s, nodes_per_sec, ...
        if length(toks) < 7
            continue
        end
        iter_local = try parse(Int, toks[1]) catch; continue end
        gap_rel = try parse(Float64, toks[6]) catch; continue end
        time_s = try parse(Float64, toks[7]) catch; continue end
        iter_global = iter_offset + iter_local
        last_iter_in_block = max(last_iter_in_block, iter_local)
        push!(it, iter_global)
        push!(t, time_s)
        push!(g, gap_rel)
    end
    df = DataFrame(iter = it, time_s = t, rel_gap = g)
    # These logs restart time at each Boscia Algorithm block; enforce monotone time for plotting.
    df = sort(df, [:time_s, :iter])
    df
end

function cb_path(criterion::String, n::Int, seed::Int, variant::String)::String
    # Variants are as they appear in filenames.
    # Some older files use "exlcusion_criterion" typo; we support both by probing.
    base = joinpath(ROOT, @sprintf("cb_%s_Boscia_%d_IND_%d_%s_one", criterion, n, seed, variant))
    # Find matching file with any job id suffix.
    # We avoid glob tool here; in Julia we can scan directory.
    for f in readdir(ROOT)
        if startswith(f, basename(base)) && endswith(f, ".txt")
            return joinpath(ROOT, f)
        end
    end
    return ""
end

function plot_one(n::Int, seed::Int; save::Bool = true)
    # Color-blind friendly palette (from colours.jl)
    rgb(t::Tuple{<:Real,<:Real,<:Real}) = RGB(float(t[1]), float(t[2]), float(t[3]))
    # Give every line a distinct color (no reuse), relying on the CB palette.
    col_acst_default = rgb(cb_blue)
    col_acst_rank = rgb(cb_purple)
    col_acst_eigen = rgb(cb_green_sea)
    col_acsts_default = rgb(cb_rose)
    col_acsts_rank = rgb(cb_blue_green)
    col_acsts_eigen = rgb(cb_clay)
    col_acsts_excl = rgb(cb_salmon_pink)
    series = [
        ("ACST default", "ACST", "baseline", :solid, col_acst_default),
        ("ACST rank-prune", "ACST", "rank_based_pruning", :dot, col_acst_rank),
        ("ACST eigenvalue-prune", "ACST", "eigenvalue_based_pruning", :dashdot, col_acst_eigen),
        ("ACSTS default", "ACSTS", "baseline", :dash, col_acsts_default),
        ("ACSTS rank-prune", "ACSTS", "rank_based_pruning", :dash, col_acsts_rank),
        ("ACSTS eigenvalue-prune", "ACSTS", "eigenvalue_based_pruning", :dashdot, col_acsts_eigen),
        ("ACSTS exclusion", "ACSTS", "exclusion_criterion", :dot, col_acsts_excl),
    ]

    # Some ACSTS runs were saved with typo "exlcusion_criterion"
    alt_excl = "exlcusion_criterion"

    p_nodes = plot(;
        xaxis = :log,
        yaxis = :log,
        xlabel = "iteration",
        ylabel = "relative gap",
        legend = :topright,
        fontfamily = "Computer Modern",
        guidefontsize = 12,
        tickfontsize = 10,
        legendfontsize = 9,
    )

    any_added = false
    for (label, crit, variant, ls, col) in series
        path = cb_path(crit, n, seed, variant)
        if isempty(path) && crit == "ACSTS" && variant == "exclusion_criterion"
            path = cb_path(crit, n, seed, alt_excl)
        end
        if isempty(path)
            @warn "Missing callback log" crit=crit n=n seed=seed variant=variant
            continue
        end
        df = parse_boscia_gap_table(path)
        if nrow(df) == 0
            @warn "No table rows parsed" path=path
            continue
        end
        plot!(p_nodes, df.iter, df.rel_gap; label = label, linestyle = ls, color = col, linewidth = 2, alpha = 0.9)
        any_added = true
    end

    if !any_added
        @warn "No series plotted" n=n seed=seed
        return nothing
    end

    fig = plot(p_nodes; size = (700, 540), left_margin = 22Plots.mm, bottom_margin = 18Plots.mm)

    if save
        mkpath(OUT_DIR)
        # Paper figure (PDF); keep PNG for quick preview.
        out_pdf = joinpath(OUT_DIR, @sprintf("acst_acsts_relgap_nodes_n%d_seed%d.pdf", n, seed))
        out_png = joinpath(OUT_DIR, @sprintf("acst_acsts_relgap_nodes_n%d_seed%d.png", n, seed))
        savefig(fig, out_pdf)
        savefig(fig, out_png)
        # Also stage next to other Computer Modern paper figures.
        paper_dir = joinpath(ROOT, "..", "plot")
        if isdir(paper_dir)
            cp(out_pdf, joinpath(paper_dir, basename(out_pdf)); force = true)
            cp(out_png, joinpath(paper_dir, basename(out_png)); force = true)
        end
        @info "Saved" out_pdf out_png
    end
    return fig
end

function main()
    # Default paper figure: n=10, seed=1. Also regenerate the other preview panels.
    for n in (10, 12)
        for seed in (1, 2, 3)
            plot_one(n, seed)
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

