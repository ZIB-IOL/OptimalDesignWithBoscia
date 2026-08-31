using Plots.PlotMeasures: mm

const PAPER_DIR = "/Users/deborah/Documents/research_projects/Smoothing-in-Boscia/paper"

"""Keyword arguments for Plots.jl figures that match LaTeX Computer Modern."""
function cm_plot_kwargs(;
    guidefontsize=11,
    tickfontsize=10,
    legendfontsize=10,
    titlefontsize=11,
    left_margin=20mm,
    right_margin=8mm,
    top_margin=8mm,
    bottom_margin=16mm,
)
    return (
        fontfamily="Computer Modern",
        guidefontsize=guidefontsize,
        tickfontsize=tickfontsize,
        legendfontsize=legendfontsize,
        titlefontsize=titlefontsize,
        left_margin=left_margin,
        right_margin=right_margin,
        top_margin=top_margin,
        bottom_margin=bottom_margin,
    )
end

function copy_to_paper(src::AbstractString)
    isfile(src) || return false
    mkpath(PAPER_DIR)
    dst = joinpath(PAPER_DIR, basename(src))
    cp(src, dst; force=true)
    println("Copied $(basename(src)) -> paper/")
    return true
end
