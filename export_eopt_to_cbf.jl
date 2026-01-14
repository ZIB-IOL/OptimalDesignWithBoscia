#!/usr/bin/env julia

"""
Script to export E-optimal design models from Pajarito to CBF (Conic Benchmark Format) files.

This script generates E-optimal design models for various dimensions and random seeds,
then exports them to CBF format for solving with external solvers.

Usage: julia export_eopt_to_cbf.jl
"""

using JuMP
using Pajarito
using MathOptInterface
using Random
using Printf
using LinearAlgebra
using Distributions
using StableRNGs
const MOI = MathOptInterface

# Include the ODWB module functions
include("ODWB/src/ODWB.jl")
using .ODWB
# Import specific functions we need
import .ODWB: build_data, build_integer_data

"""
    create_eopt_cbf_model(seed, m, n, criterion, corr; integer_data=false, zero_one=false, N=-Inf)

Create an E-optimal design model suitable for CBF export.
This function creates a model without solver-specific configurations.

# Arguments
- `seed`: Random seed for reproducibility
- `m`: Number of experiments
- `n`: Number of parameters  
- `criterion`: "E" for standard E-optimal, "EF" for fusion E-optimal
- `corr`: Boolean for correlated vs independent data
- `integer_data`: Boolean for integer vs continuous data generation
- `zero_one`: Boolean for 0-1 constraints
- `N`: Number of experiments to run (if -Inf, uses default)

# Returns
- `model`: JuMP model ready for CBF export
- `x`: Decision variables
- `t`: Objective variable
- `A`: Design matrix
- `C`: Fusion matrix (if applicable)
- `ub`: Upper bounds
"""
function create_eopt_cbf_model(seed, m, n, criterion, corr; integer_data=false, zero_one=false, N=-Inf)
    # Generate data using ODWB utilities
    if criterion == "EF"
        A, C, N, ub, _ = integer_data ? build_integer_data(seed, m, n, true, corr, zero_one=zero_one, N=N) : build_data(seed, m, n, true, corr, zero_one=zero_one, N=N)
    else
        A, _, N, ub, _ = integer_data ? build_integer_data(seed, m, n, false, corr, zero_one=zero_one, N=N) : build_data(seed, m, n, false, corr, zero_one=zero_one, N=N)
        C = nothing
    end

    # Create model without solver (for CBF export)
    model = Model()
    
    # Add variables
    @variable(model, x[1:m])
    for i in 1:m
        set_integer(x[i])
    end
    @variable(model, t)
    
    # Add constraints - use inequalities for better CBF compatibility
    @constraint(model, sum(x) >= N)
    @constraint(model, sum(x) <= N)
    @constraint(model, x >= 0)
    @constraint(model, x <= ub)
    
    # Add E-optimal constraint: A' * diag(x) * A + t*I ⪰ 0
    if criterion == "E"
        # Information matrix: A' * diag(x) * A + t*I
        info_matrix = [
            @expression(model, 
                (i == j ? -t : 0.0) + sum(A[k, i] * x[k] * A[k, j] for k in 1:m)
            ) for i in 1:n, j in 1:n
        ]
        # Add PSD constraint
        @constraint(model, info_matrix in PSDCone())
    elseif criterion == "EF"
        # For fusion case, use C matrix as well
        info_matrix = [
            @expression(model, 
                C[i, j] + (i == j ? -t : 0.0) + sum(A[k, i] * x[k] * A[k, j] for k in 1:m)
            ) for i in 1:n, j in 1:n
        ]
        # Add PSD constraint  
        @constraint(model, info_matrix in PSDCone())
    end
    
    # Set objective (maximize t)
    @objective(model, Max, t)

    return model, x, t, A, C, N, ub
end

"""
    export_model_to_cbf(model, filename)

Export a JuMP model to CBF format.

# Arguments
- `model`: JuMP model to export
- `filename`: Output CBF filename
"""
function export_model_to_cbf(model, filename)
    try
        # Create a FileFormats model and copy the JuMP model to it
        cbf_model = MOI.FileFormats.Model(format = MOI.FileFormats.FORMAT_CBF)
        
        # Use bridges to handle unsupported constraints
        bridged_model = MOI.Bridges.full_bridge_optimizer(cbf_model, Float64)
        MOI.copy_to(bridged_model, backend(model))
        
        # Write to CBF file
        MOI.write_to_file(cbf_model, filename)
        println("✓ Successfully exported model to: $filename")
        return true
    catch e
        println("✗ Error exporting model to $filename: $e")
        println("  This might be due to unsupported constraint types in CBF format")
        return false
    end
end

"""
    generate_cbf_files(dimensions, seeds, criteria, correlations; output_dir="cbf_models", integer_data=false, zero_one=false)

Generate CBF files for multiple combinations of parameters.

# Arguments
- `dimensions`: Vector of (m, n) tuples for different problem sizes
- `seeds`: Vector of random seeds
- `criteria`: Vector of criteria ("E", "EF")
- `correlations`: Vector of correlation flags (true/false)
- `output_dir`: Base directory for output files
- `integer_data`: Use integer data generation
- `zero_one`: Use 0-1 constraints
"""
function generate_cbf_files(dimensions, seeds, criteria, correlations; output_dir="cbf_models", integer_data=false, zero_one=false)
    # Create base output directory
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
                    
                    # Create descriptive filename
                    corr_str = corr ? "corr" : "indep"
                    data_str = integer_data ? "int" : "cont"
                    zero_one_str = zero_one ? "_01" : ""
                    
                    filename = @sprintf("eopt_%s_%s%s_m%d_n%d_seed%d.cbf", 
                                      criterion, corr_str, zero_one_str, m, n, seed)
                    filepath = joinpath(output_dir, filename)
                    
                    println("[$current_model/$total_models] Generating: $filename")
                    
                    try
                        # Create model
                        model, x_vars, t_var, A, C, N, ub = create_eopt_cbf_model(
                            seed, m, n, criterion, corr, 
                            integer_data=integer_data, zero_one=zero_one
                        )
                        
                        # Export to CBF
                        success = export_model_to_cbf(model, filepath)
                        
                        if success
                            # Print model info
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
    main()

Main function to run the CBF export script.
"""
function main()
    println("E-Optimal Design CBF Export Script")
    println("="^40)
    
    # Configuration parameters
    num_experiments = [50, 80, 100, 120, 150]
    # Option 1: Create (m, n) pairs where n = floor(sqrt(m))
    dimensions = [(m, Int(floor(sqrt(m)))) for m in num_experiments]
    
    seeds = [1, 2, 3, 4, 5]  # Multiple random seeds
    criteria = ["E"] # "EF"  # Both standard and fusion E-optimal
    correlations = [false, true]  # Both independent and correlated data
    
    # Output directory
    output_dir = "cbf_eopt_models"
    
    # Generate CBF files
    generate_cbf_files(
        dimensions, seeds, criteria, correlations,
        output_dir=output_dir,
        integer_data=false,  # Use continuous data generation
        zero_one=false       # Don't use 0-1 constraints
    )
    
    # Also generate some integer data versions for comparison
    println("\nGenerating integer data versions...")
    generate_cbf_files(
        dimensions[1:2], seeds[1:3], criteria, [false],  # Smaller subset for integer data
        output_dir=joinpath(output_dir, "integer_data"),
        integer_data=true,
        zero_one=false
    )
    
    println("\nAll CBF files generated successfully!")
    println("You can now solve these models with any CBF-compatible solver.")
end

# Run the script if called directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
