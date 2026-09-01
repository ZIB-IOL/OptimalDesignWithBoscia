#!/usr/bin/env julia
#
# Root-node upper-bound scatter: Boscia vs SCIP-SDP (optimized preset).
# Recreates scatter_scip_vs_boscia_optimized.pdf from root_lb_pairs.csv.

using CSV
using DataFrames
using Plots
using Plots.PlotMeasures: mm
include(joinpath(@__DIR__, "..", "plot", "plot_style.jl"))
include(joinpath(@__DIR__, "colours.jl"))

const CSV_PATH = joinpath(@__DIR__, "plots", "root_lb", "root_lb_pairs.csv")
const OUT_DIR = joinpath(@__DIR__, "plots", "root_lb")

rgb(t) = RGB(t[1], t[2], t[3])

const SCIP_BNB_COLOR = rgb(cb_green_sea)
const SCIP_OA_COLOR = rgb(cb_salmon_pink)

function ub_value(x)
    x === missing && return NaN
    x isa Number || return NaN
    isfinite(x) || return NaN
    return -Float64(x)
end

function panel_data(df::DataFrame, crit::String, typ::String)
    sub = df[(df.crit .== crit) .& (df.typ .== typ) .& (df.folder .== "optimized"), :]
    boscia = [ub_value(x) for x in sub.bos_lb]
    bnb = [ub_value(x) for x in sub.bnb_lb]
    oa = [ub_value(x) for x in sub.oa_lb]
    mask_bnb = isfinite.(boscia) .& isfinite.(bnb)
    mask_oa = isfinite.(boscia) .& isfinite.(oa)
    return (
        boscia=boscia,
        bnb=bnb,
        oa=oa,
        mask_bnb=mask_bnb,
        mask_oa=mask_oa,
        n_bnb=sum(mask_bnb),
        n_oa=sum(mask_oa),
    )
end

function panel_limits(data)
    xs = data.boscia
    all_x = vcat(xs[data.mask_bnb], xs[data.mask_oa])
    all_y = vcat(data.bnb[data.mask_bnb], data.oa[data.mask_oa])
    isempty(all_x) && return (0.0, 1.0)
    lo = min(minimum(all_x), minimum(all_y))
    hi = max(maximum(all_x), maximum(all_y))
    span = hi - lo
    span <= 0 && return (lo - 1.0, hi + 1.0)
    pad = 0.02 * span
    lo -= pad
    hi += pad
    if minimum(vcat(all_x, all_y)) > 0
        lo = max(0, lo)
    end
    return (lo, hi)
end

function add_panel!(plt, panel_idx::Int, data, title::String; show_legend::Bool=false)
    lims = panel_limits(data)
    xs = data.boscia
    hide = (label=false, legend_entry=false)

    plot!(
        plt,
        [lims[1], lims[2]],
        [lims[1], lims[2]];
        subplot=panel_idx,
        label=show_legend ? "equal UB" : false,
        legend_entry=show_legend,
        color=:gray,
        linestyle=:dash,
        linewidth=1.5,
    )
    scatter!(
        plt,
        xs[data.mask_bnb],
        data.bnb[data.mask_bnb];
        subplot=panel_idx,
        label=show_legend ? "SCIPSDP BnB" : false,
        legend_entry=show_legend,
        color=SCIP_BNB_COLOR,
        marker=:utriangle,
        markersize=5,
        markerstrokewidth=0,
    )
    scatter!(
        plt,
        xs[data.mask_oa],
        data.oa[data.mask_oa];
        subplot=panel_idx,
        label=show_legend ? "SCIPSDP OA" : false,
        legend_entry=show_legend,
        color=SCIP_OA_COLOR,
        marker=:rect,
        markersize=5,
        markerstrokewidth=0,
    )
    plot!(
        plt,
        subplot=panel_idx,
        title=title,
        xlabel="Boscia root upper bound",
        ylabel="SCIP root upper bound",
        xlims=lims,
        ylims=lims,
        aspect_ratio=1,
        grid=true;
        hide...,
    )
end

function main()
    isfile(CSV_PATH) || error("Missing $CSV_PATH")
    df = CSV.read(CSV_PATH, DataFrame)

    panels = [
        ("E", "independent", "E independent"),
        ("E", "correlated", "E correlated"),
        ("AGC", "independent", "AGC independent"),
        ("AGC", "correlated", "AGC correlated"),
    ]

    plt = plot(
        layout=(2, 2),
        size=(1100, 1100),
        legend=:topleft,
        legend_subplot=1;
        cm_plot_kwargs(
            guidefontsize=12,
            tickfontsize=10,
            legendfontsize=10,
            left_margin=8mm,
            right_margin=4mm,
            top_margin=4mm,
            bottom_margin=8mm,
        )...
    )
    for (idx, (crit, typ, title)) in enumerate(panels)
        add_panel!(plt, idx, panel_data(df, crit, typ), title; show_legend=(idx == 1))
    end

    mkpath(OUT_DIR)
    out = joinpath(OUT_DIR, "scatter_scip_vs_boscia_optimized.pdf")
    savefig(plt, out)
    println("Wrote $out")
    copy_to_paper(out)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
