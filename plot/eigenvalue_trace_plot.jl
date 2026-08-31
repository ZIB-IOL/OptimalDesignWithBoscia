using CSV
using DataFrames
using Plots
using Plots.PlotMeasures: mm

include(joinpath(@__DIR__, "..", "ODWB", "colours.jl"))
include(joinpath(@__DIR__, "plot_style.jl"))

const EIGENVALUE_DIR = joinpath(@__DIR__, "..", "ODWB", "csv", "Boscia", "eigenvalue_list")
const OUT_DIR = @__DIR__

rgb_hex(c::NTuple{3, <:Real}) = "#" * join((uppercase(string(v, base=16, pad=2)) for v in round.(Int, 255 .* collect(c))), "")

const CB_EIGEN_COLORS = [
    rgb_hex(cb_blue),
    rgb_hex(cb_burgundy),
    rgb_hex(cb_green_lime),
    rgb_hex(cb_purple),
    rgb_hex(cb_salmon_pink),
    rgb_hex(cb_black),
]
const EIGEN_LINESTYLES = [:solid, :dash, :dot, :dashdot, :solid, :dash]
const EIGEN_MARKERS = [:circle, :rect, :utriangle, :diamond, :star5, :xcross]

function parse_eigenvalue_string(s::AbstractString)
    stripped = strip(s)
    startswith(stripped, "[") || error("Unexpected eigenvalue format: $s")
    endswith(stripped, "]") || error("Unexpected eigenvalue format: $s")
    body = stripped[2:end-1]
    isempty(strip(body)) && return Float64[]
    return [parse(Float64, strip(part)) for part in split(body, ",")]
end

function load_eigenvalue_matrix(path::String)
    df = DataFrame(CSV.File(path))
    hasproperty(df, :eigenvalue) || error("Missing eigenvalue column in $path")
    rows = [parse_eigenvalue_string(row.eigenvalue) for row in eachrow(df)]
    isempty(rows) && error("No eigenvalue rows in $path")
    nλ = length(rows[1])
    all(length(vals) == nλ for vals in rows) || error("Inconsistent eigenvalue lengths in $path")
    return reduce(vcat, permutedims.(rows))
end

function load_actual_smallest_eigenvalue(path::String)
    df = DataFrame(CSV.File(path; delim=';'))
    nrow(df) == 0 && error("No rows in $path")
    row = df[end, :]
    hasproperty(row, :scaled_solution) || error("Missing scaled_solution in $path")
    return -Float64(row.scaled_solution)
end

function add_panel!(plt, eigvals::Matrix{Float64}, panel_idx::Int; point_stride::Int=1)
    n_steps, nλ = size(eigvals)
    x = collect(1:point_stride:n_steps)
    for i in 1:nλ
        plot!(
            plt,
            x,
            eigvals[1:point_stride:end, i],
            subplot=panel_idx,
            label="λ_$i",
            color=CB_EIGEN_COLORS[i],
            linestyle=EIGEN_LINESTYLES[i],
            marker=EIGEN_MARKERS[i],
            markersize=3,
            markevery=max(cld(n_steps, 12), 1),
            linewidth=2,
        )
    end
    plot!(
        plt,
        subplot=panel_idx,
        xlabel="nodes",
        ylabel=panel_idx == 1 ? "Eigenvalue" : "",
        grid=true,
        legend=:topright,
    )
end

function plot_eigenvalue_traces(independent_path::String, correlated_path::String; out_path::Union{Nothing,String}=nothing)
    independent = load_eigenvalue_matrix(independent_path)
    correlated = load_eigenvalue_matrix(correlated_path)
    size(independent, 2) == size(correlated, 2) || error("Different numbers of eigenvalues between files")

    plt = plot(
        layout=(1, 2),
        size=(1200, 520);
        cm_plot_kwargs(guidefontsize=10, tickfontsize=8, legendfontsize=8)...
    )
    add_panel!(plt, correlated, 1)
    add_panel!(plt, independent, 2; point_stride=10)

    output = something(out_path, joinpath(OUT_DIR, "E_eigenvalue_trace_30_5_7_1.pdf"))
    savefig(plt, output)
    println("Wrote $output")
    copy_to_paper(output)
    return output
end

function plot_smallest_eigenvalue_trace(independent_path::String, correlated_path::String; out_path::Union{Nothing,String}=nothing)
    independent = load_eigenvalue_matrix(independent_path)
    correlated = load_eigenvalue_matrix(correlated_path)
    independent_solution = load_actual_smallest_eigenvalue(replace(independent_path, "eigenvalue_list/" => ""))
    correlated_solution = load_actual_smallest_eigenvalue(replace(correlated_path, "eigenvalue_list/" => ""))

    plt = plot(
        layout=(1, 2),
        size=(1200, 520);
        cm_plot_kwargs(guidefontsize=10, tickfontsize=8, legendfontsize=8)...
    )
    plot!(
        plt,
        1:size(correlated, 1),
        correlated[:, 1],
        subplot=1,
        label="λ_1",
        color=CB_EIGEN_COLORS[1],
        linewidth=2,
        xlabel="nodes",
        ylabel="λ_1",
        grid=true,
        legend=false,
    )
    hline!(
        plt,
        [correlated_solution],
        subplot=1,
        label="actual solution",
        color=rgb_hex(cb_black),
        linestyle=:dash,
        linewidth=2,
    )
    plot!(
        plt,
        1:size(independent, 1),
        independent[:, 1],
        subplot=2,
        label="λ_1",
        color=CB_EIGEN_COLORS[2],
        linewidth=2,
        xlabel="nodes",
        ylabel="",
        grid=true,
        legend=false,
    )
    hline!(
        plt,
        [independent_solution],
        subplot=2,
        label="actual solution",
        color=rgb_hex(cb_black),
        linestyle=:dash,
        linewidth=2,
    )

    output = something(out_path, joinpath(OUT_DIR, "E_smallest_eigenvalue_trace_30_5_7_1.pdf"))
    savefig(plt, output)
    println("Wrote $output")
    copy_to_paper(output)
    return output
end

function main(args=ARGS)
    independent_path = joinpath(EIGENVALUE_DIR, "boscia__E_optimality_independent__30_5_7_1.csv")
    correlated_path = joinpath(EIGENVALUE_DIR, "boscia__E_optimality_correlated__30_5_7_1.csv")
    out_path = nothing
    if length(args) >= 2
        independent_path = args[1]
        correlated_path = args[2]
    end
    if length(args) >= 3
        out_path = args[3]
    end
    plot_eigenvalue_traces(independent_path, correlated_path; out_path)
    plot_smallest_eigenvalue_trace(independent_path, correlated_path)
end

main()
