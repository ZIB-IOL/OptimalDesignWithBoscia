#!/usr/bin/env julia
#
# Regenerate paper figures with Computer Modern and copy to Smoothing-in-Boscia/paper/.
#
# Usage (from OptimalDesignWithBoscia):
#   julia --project=. plot/regenerate_paper_figures.jl

const ROOT = dirname(@__DIR__)
const PLOT_DIR = joinpath(ROOT, "plot")
const PAPER_DIR = "/Users/deborah/Documents/research_projects/Smoothing-in-Boscia/paper"

include(joinpath(PLOT_DIR, "plot_style.jl"))

const PAPER_TERMINATION_PLOTS = [
    "E_correlated_scipsdp.pdf",
    "E_independent_scipsdp.pdf",
    "AGC_correlated_scipsdp.pdf",
    "AGC_independent_scipsdp.pdf",
    "E_correlated_pruning.pdf",
    "E_independent_pruning.pdf",
    "AGC_correlated_pruning.pdf",
    "AGC_independent_pruning.pdf",
    "ACST_independent_pruning.pdf",
    "AGC_correlated_exclusion.pdf",
]

function run_script(rel_path::String, args=String[])
    script = joinpath(ROOT, rel_path)
    cmd = Cmd(["julia", "--project=$(ROOT)", script, args...])
    println("\n>>> ", join(cmd.exec, " "))
    run(cmd)
end

function copy_termination_plots()
    src_dir = joinpath(PLOT_DIR, "termination_plots")
    for fname in PAPER_TERMINATION_PLOTS
        copy_to_paper(joinpath(src_dir, fname))
    end
end

function main()
    println("Regenerating paper figures into: ", PAPER_DIR)

    run_script("plot/plot.jl", ["--families", "scipsdp,pruning,exclusion"])
    copy_termination_plots()

    run_script("ODWB/plot_longrun_relgap.jl")

    run_script("plot/eigenvalue_trace_plot.jl")

    run_script("ODWB/plot_root_lb_scatter.jl")

    println("\nDone. Trajectory-only figures were not regenerated (see script output / README).")
end

main()
