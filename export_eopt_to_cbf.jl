#!/usr/bin/env julia

"""
Export JuMP models to CBF (Conic Benchmark Format) for external solvers (e.g. SCIP-SDP).
show
Two families:
- **E-optimal design** (`eopt`): PSD information-matrix constraints from `build_data` / `build_integer_data`.
- **ACST** (algebraic connectivity spanning tree): LaplacianOpt JSON instances via
  `ODWB.algebraic_connectivity_model` (same as `solve_opt_scip_sdp` for criterion ACST).

Usage (from repo root):

    julia --project=ODWB export_eopt_to_cbf.jl              # E-optimal only (default)
    julia --project=ODWB export_eopt_to_cbf.jl eopt
    julia --project=ODWB export_eopt_to_cbf.jl acst
    julia --project=ODWB export_eopt_to_cbf.jl all         # eopt then acst

E-opt output: `cbf_eopt_models/` (and `cbf_eopt_models/integer_data/`).
ACST output: `ODWB/cbf_eopt_models/acst/`.

When ODWB is loaded via `include` (not as a package), `LAPLACIANOPT_INSTANCES_ROOT` must be set
for ACST; this script sets it to `ODWB/data/laplacianopt_instances` before loading ODWB.
"""

# LaplacianOpt JSON path (must be set before `include` ODWB for ACST / `laplacianopt_instance_file`)
const _LAP_INST_ROOT = joinpath(@__DIR__, "ODWB", "data", "laplacianopt_instances")
ENV["LAPLACIANOPT_INSTANCES_ROOT"] = _LAP_INST_ROOT

using JuMP
using Pajarito
using MathOptInterface
using Random
using Printf
using LinearAlgebra
using Distributions
using StableRNGs
const MOI = MathOptInterface

include(joinpath(@__DIR__, "ODWB", "src", "ODWB.jl"))
using .ODWB
import .ODWB: build_data, build_integer_data

function create_eopt_cbf_model(seed, m, n, criterion, corr; integer_data=false, zero_one=false, N=-Inf)
    if criterion == "EF"
        A, C, N, ub, _ = integer_data ? build_integer_data(seed, m, n, true, corr, zero_one=zero_one, N=N) : build_data(seed, m, n, true, corr, zero_one=zero_one, N=N)
    else
        A, _, N, ub, _ = integer_data ? build_integer_data(seed, m, n, false, corr, zero_one=zero_one, N=N) : build_data(seed, m, n, false, corr, zero_one=zero_one, N=N)
        C = nothing
    end

    model = Model()
    @variable(model, x[1:m])
    for i in 1:m
        set_integer(x[i])
    end
    @variable(model, t)
    @constraint(model, sum(x) >= N)
    @constraint(model, sum(x) <= N)
    @constraint(model, x >= 0)
    @constraint(model, x <= ub)

    if criterion == "E"
        info_matrix = [
            @expression(model,
                (i == j ? -t : 0.0) + sum(A[k, i] * x[k] * A[k, j] for k in 1:m)
            ) for i in 1:n, j in 1:n
        ]
        @constraint(model, info_matrix in PSDCone())
    elseif criterion == "EF"
        info_matrix = [
            @expression(model,
                C[i, j] + (i == j ? -t : 0.0) + sum(A[k, i] * x[k] * A[k, j] for k in 1:m)
            ) for i in 1:n, j in 1:n
        ]
        @constraint(model, info_matrix in PSDCone())
    end
    @objective(model, Max, t)

    return model, x, t, A, C, N, ub
end

function export_model_to_cbf(model, filename; quiet_success::Bool=false)
    try
        cbf_model = MOI.FileFormats.Model(format = MOI.FileFormats.FORMAT_CBF)
        bridged_model = MOI.Bridges.full_bridge_optimizer(cbf_model, Float64)
        MOI.copy_to(bridged_model, backend(model))
        MOI.write_to_file(cbf_model, filename)
        quiet_success || println("✓ Successfully exported model to: $filename")
        return true
    catch e
        println("✗ Error exporting model to $filename: $e")
        println("  This might be due to unsupported constraint types in CBF format")
        return false
    end
end

function generate_cbf_files(dimensions, seeds, criteria, correlations; output_dir="cbf_models", integer_data=false, zero_one=false)
    mkpath(output_dir)
    total_models = length(dimensions) * length(seeds) * length(criteria) * length(correlations)
    current_model = 0
    println("Generating $total_models CBF models...")
    println("Output directory: $output_dir")
    println("="^60)

    for (m, n) in dimensions
        for seed in seeds
            for criterion in criteria
                for corr in correlations
                    current_model += 1
                    corr_str = corr ? "corr" : "indep"
                    zero_one_str = zero_one ? "_01" : ""
                    filename = @sprintf("eopt_%s_%s%s_m%d_n%d_seed%d.cbf",
                        criterion, corr_str, zero_one_str, m, n, seed)
                    filepath = joinpath(output_dir, filename)
                    println("[$current_model/$total_models] Generating: $filename")
                    try
                        model, x_vars, t_var, A, C, N, ub = create_eopt_cbf_model(
                            seed, m, n, criterion, corr,
                            integer_data=integer_data, zero_one=zero_one
                        )
                        success = export_model_to_cbf(model, filepath)
                        if success
                            println("  Parameters: m=$m, n=$n, N=$N, criterion=$criterion")
                            println("  Variables: $(num_variables(model))")
                            println("  Upper bounds range: [$(minimum(ub)), $(maximum(ub))]")
                        end
                    catch e
                        println("✗ Error creating model for $filename: $e")
                        continue
                    end
                    println()
                end
            end
        end
    end
    println("="^60)
    println("CBF file generation complete!")
    println("Files saved in: $output_dir")
end

"""
    export_acst_cbfs(; output_dir, node_instance_pairs, use_base_graph=false)

`node_instance_pairs`: `(n_nodes, instance_id)` matching
`ODWB/data/laplacianopt_instances/{n}_nodes/{n}_{instance}.json`.
"""
function export_acst_cbfs(;
    output_dir = joinpath(@__DIR__, "ODWB", "cbf_eopt_models", "acst"),
    node_instance_pairs = [(5, i) for i in 1:5],
    use_base_graph::Bool = false,
)
    mkpath(output_dir)
    base_suffix = use_base_graph ? "_basegraph" : ""
    println("Exporting $(length(node_instance_pairs)) ACST models to: $output_dir")
    for (n_nodes, instance) in node_instance_pairs
        inst_path = joinpath(_LAP_INST_ROOT, "$(n_nodes)_nodes", "$(n_nodes)_$(instance).json")
        if !isfile(inst_path)
            println("  skip missing: $inst_path")
            continue
        end
        m_dummy = Int(n_nodes * (n_nodes - 1) / 2)
        model, _, _ = ODWB.algebraic_connectivity_model(
            instance,
            m_dummy,
            n_nodes;
            build_spanning_tree = true,
            use_base_graph = use_base_graph,
            augment_budget = -1,
        )
        fname = "acst_n$(n_nodes)_inst$(instance)$(base_suffix).cbf"
        fpath = joinpath(output_dir, fname)
        try
            if export_model_to_cbf(model, fpath; quiet_success = true)
                println("  ✓ $fname  (variables: $(num_variables(model)))")
            else
                println("  ✗ $fname  (export failed)")
            end
        catch e
            println("  ✗ $fname : $e")
        end
    end
    println("ACST export done.")
end

function main_eopt()
    println("E-optimal design CBF export")
    println("="^40)
    num_experiments = [50, 80, 100, 120, 150]
    dimensions = [(m, Int(floor(sqrt(m)))) for m in num_experiments]
    seeds = [1, 2, 3, 4, 5]
    criteria = ["E"]
    correlations = [false, true]
    output_dir = "cbf_eopt_models"
    generate_cbf_files(
        dimensions, seeds, criteria, correlations;
        output_dir = output_dir,
        integer_data = false,
        zero_one = false,
    )
    println("\nGenerating integer data versions...")
    generate_cbf_files(
        dimensions[1:2], seeds[1:3], criteria, [false];
        output_dir = joinpath(output_dir, "integer_data"),
        integer_data = true,
        zero_one = false,
    )
    println("\nE-opt CBF generation finished.")
end

function main_acst()
    println("ACST (LaplacianOpt) CBF export")
    println("="^40)
    pairs = vcat(
        [(5, i) for i in 1:5],
        [(8, i) for i in 1:3],
        [(10, i) for i in 1:3],
    )
    export_acst_cbfs(; node_instance_pairs = pairs, use_base_graph = false)
end

function main()
    mode = isempty(ARGS) ? "eopt" : lowercase(ARGS[1])
    if mode == "eopt"
        main_eopt()
    elseif mode == "acst"
        main_acst()
    elseif mode == "all"
        main_eopt()
        println()
        main_acst()
    else
        error("Unknown mode $(repr(mode)). Use: eopt (default), acst, or all.")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
