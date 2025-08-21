#!/usr/bin/env julia

"""
Batch script to generate progress plots for all available dimension and seed combinations
from the Boscia full runs data.

Usage: julia generate_all_progress_plots.jl
"""

using Printf

function parse_filename(filename)
    """Parse filename to extract dimension, seed, and data type."""
    base_name = replace(filename, ".csv" => "")
    
    # Extract data type
    data_type = if contains(base_name, "_cont__")
        "cont"
    elseif contains(base_name, "_int__")
        "int"
    else
        "unknown"
    end
    
    # Extract dimension and seed from the end pattern: __<dimension>_<param>_<seed>
    parts = split(base_name, "_")
    if length(parts) >= 3
        try
            seed = parse(Int, parts[end])
            dimension = parse(Int, parts[end-2])
            return (dimension=dimension, seed=seed, data_type=data_type)
        catch
            return nothing
        end
    end
    
    return nothing
end

function find_all_combinations()
    """Find all available dimension, seed, and data type combinations."""
    
    full_runs_dir = "ODWB/csv/full_runs_boscia"
    
    if !isdir(full_runs_dir)
        error("Directory not found: $full_runs_dir")
    end
    
    all_files = readdir(full_runs_dir)
    csv_files = filter(f -> endswith(f, ".csv"), all_files)
    
    combinations = []
    
    for file in csv_files
        file_info = parse_filename(file)
        if file_info !== nothing && file_info.data_type != "unknown"
            push!(combinations, (
                dimension = file_info.dimension,
                seed = file_info.seed,
                data_type = file_info.data_type,
                filename = file
            ))
        end
    end
    
    return sort(combinations, by = x -> (x.dimension, x.seed, x.data_type))
end

function generate_all_plots()
    """Generate progress plots for all available combinations."""
    
    println("=== Batch Progress Plot Generation ===")
    println()
    
    # Find all combinations
    combinations = find_all_combinations()
    
    if isempty(combinations)
        println("No valid combinations found!")
        return
    end
    
    println("Found $(length(combinations)) combinations to plot:")
    for combo in combinations
        println("  Dimension $(combo.dimension), Seed $(combo.seed), Type $(combo.data_type)")
    end
    println()
    
    # Ensure plots directory exists
    plots_dir = "plots"
    if !isdir(plots_dir)
        mkdir(plots_dir)
        println("Created plots directory: $plots_dir")
    end
    
    # Generate plots for each combination
    successful_plots = 0
    failed_plots = 0
    
    for (i, combo) in enumerate(combinations)
        print("[$i/$(length(combinations))] Generating plot for D$(combo.dimension)_S$(combo.seed)_$(combo.data_type)...")
        
        try
            # Run the plotting script for this combination
            cmd = `julia plot_boscia_progress.jl $(combo.dimension) $(combo.seed) $(combo.data_type)`
            result = run(pipeline(cmd, stdout=devnull, stderr=devnull))
            
            if result.exitcode == 0
                println(" ✓ Success")
                successful_plots += 1
            else
                println(" ✗ Failed (exit code: $(result.exitcode))")
                failed_plots += 1
            end
            
        catch e
            println(" ✗ Error: $e")
            failed_plots += 1
        end
    end
    
    # Summary
    println()
    println("=== Generation Summary ===")
    println("Total combinations: $(length(combinations))")
    println("Successful plots: $successful_plots")
    println("Failed plots: $failed_plots")
    println()
    
    if successful_plots > 0
        println("Generated plots saved in: $plots_dir/")
        println("Plot naming pattern: boscia_progress_D<dimension>_S<seed>_<type>.png")
    end
    
    # List generated files
    if successful_plots > 0
        println()
        println("Generated files:")
        plot_files = filter(f -> startswith(f, "boscia_progress_") && endswith(f, ".png"), readdir(plots_dir))
        for file in sort(plot_files)
            println("  $plots_dir/$file")
        end
    end
    
    return successful_plots, failed_plots
end

function main()
    """Main function."""
    
    # Check if the plotting script exists
    if !isfile("plot_boscia_progress.jl")
        error("plot_boscia_progress.jl not found in current directory!")
    end
    
    try
        successful, failed = generate_all_plots()
        
        if failed > 0
            println("\\nSome plots failed to generate. Check the individual instances manually.")
            exit(1)
        else
            println("\\nAll plots generated successfully! 🎉")
        end
        
    catch e
        println("Error during batch generation: $e")
        exit(1)
    end
end

# Run the script
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
