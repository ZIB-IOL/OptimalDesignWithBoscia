#!/usr/bin/env julia
#
# Root-node upper-bound scatter: Boscia vs SCIP-SDP (optimized preset).
# Recreates scatter_scip_vs_boscia_optimized.pdf from root_lb_pairs.csv.

using CSV
using DataFrames
using Plots
include(joinpath(@__DIR__, "..", "plot", "plot_style.jl"))
include(joinpath(@__DIR__, "colours.jl"))

const CSV_PATH = joinpath(@__DIR__, "plots", "root_lb", "root_lb_pairs.csv")
const OUT_DIR = joinpath(@__DIR__, "plots", "root_lb")

rgb(t) = RGB(t[1], t[2], t[3])

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

function add_panel!(plt, panel_idx::Int, data, title::String)
    xs = data.boscia
    lo = minimum(vcat(xs[data.mask_bnb], xs[data.mask_oa]))
    hi = maximum(vcat(xs[data.mask_bnb], xs[data.mask_oa]))
    pad = 0.05 * (hi - lo)
    lims = (lo - pad, hi + pad)

    plot!(
        plt,
        [lims[1], lims[2]],
        [lims[1], lims[2]];
        subplot=panel_idx,
        label="equal UB",
        color=:gray,
        linestyle=:dash,
        linewidth=1.5,
        legend=:topleft,
    )
    scatter!(
        plt,
        xs[data.mask_bnb],
        data.bnb[data.mask_bnb];
        subplot=panel_idx,
        label="SCIP B&B (n=$(data.n_bnb))",
        color=rgb(cb_blue),
        marker=:circle,
        markersize=4,
    )
    scatter!(
        plt,
        xs[data.mask_oa],
        data.oa[data.mask_oa];
        subplot=panel_idx,
        label="SCIP OA (n=$(data.n_oa))",
        color=rgb(cb_salmon_pink),
        marker=:square,
        markersize=4,
    )
    plot!(
        plt,
        subplot=panel_idx,
        title=title,
        xlabel="Boscia root dual / UB",
        ylabel="SCIP root dual / UB",
        xlims=lims,
        ylims=lims,
        aspect_ratio=1,
        grid=true,
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
        size=(900, 820);
        cm_plot_kwargs(guidefontsize=12, tickfontsize=10, legendfontsize=9)...
    )
    for (idx, (crit, typ, title)) in enumerate(panels)
        add_panel!(plt, idx, panel_data(df, crit, typ), title)
    end

    mkpath(OUT_DIR)
    out = joinpath(OUT_DIR, "scatter_scip_vs_boscia_optimized.pdf")
    savefig(plt, out)
    println("Wrote $out")
    copy_to_paper(out)
end

main()
