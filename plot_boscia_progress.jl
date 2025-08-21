#!/usr/bin/env julia

"""
Progress plotting script for Boscia optimization runs.

This script plots the progress of lower and upper bounds over time and iterations
for specified instances from the full_runs_boscia data.

Usage:
    julia plot_boscia_progress.jl                                    # Interactive mode
    julia plot_boscia_progress.jl 100 1 cont                         # Plot dimension=100, seed=1, continuous
    julia plot_boscia_progress.jl 50 3 int                           # Plot dimension=50, seed=3, integer
"""

using CSV, DataFrames, PyPlot, Printf

function parse_filename(filename)
    """Parse filename to extract dimension, seed, and data type."""
    # Expected format: boscia_E_optimality_independent_[cont|int]__<dimension>_<param>_<seed>.csv
    
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

function find_matching_files(dimension, seed, data_type)
    """Find files matching the specified criteria."""
    
    full_runs_dir = "ODWB/csv/full_runs_boscia"
    
    if !isdir(full_runs_dir)
        error("Directory not found: $full_runs_dir")
    end
    
    all_files = readdir(full_runs_dir)
    csv_files = filter(f -> endswith(f, ".csv"), all_files)
    
    matching_files = []
    
    for file in csv_files
        file_info = parse_filename(file)
        if file_info !== nothing
            if (dimension === nothing || file_info.dimension == dimension) &&
               (seed === nothing || file_info.seed == seed) &&
               (data_type === nothing || file_info.data_type == data_type)
                push!(matching_files, (file=file, info=file_info))
            end
        end
    end
    
    return matching_files
end

function load_progress_data(filename)
    """Load and process progress data from a CSV file."""
    
    full_path = joinpath("ODWB/csv/full_runs_boscia", filename)
    
    if !isfile(full_path)
        error("File not found: $full_path")
    end
    
    df = CSV.read(full_path, DataFrame)
    
    # Add iteration numbers (assuming each row is an iteration)
    df.iteration = 1:nrow(df)
    
    # Convert time from milliseconds to seconds if needed
    if maximum(df.time) > 10000  # Heuristic: if max time > 10k, likely in milliseconds
        df.time_seconds = df.time ./ 1000.0
    else
        df.time_seconds = df.time
    end
    
    return df
end

function plot_progress(files_data; save_plots=true)
    """Create side-by-side plots showing bound progress over time and iterations."""
    
    if isempty(files_data)
        println("No data to plot!")
        return
    end
    
    # Create figure with subplots
    fig, axes = subplots(2, 2, figsize=(15, 10))
    
    colors = ["blue", "red", "green", "orange", "purple", "brown", "pink", "gray"]
    
    for (i, (filename, df, info)) in enumerate(files_data)
        base_color = colors[(i-1) % length(colors) + 1]
        
        # Define colors: lower bounds = base color, upper bounds = warmer tone, gaps = cooler tone
        lower_color = base_color
        upper_color = if base_color == "blue"
            "orange"
        elseif base_color == "red"
            "darkred"
        elseif base_color == "green"
            "lime"
        elseif base_color == "orange"
            "red"
        else
            "crimson"
        end
        
        gap_color = if base_color == "blue"
            "purple"
        elseif base_color == "red"
            "darkblue"
        elseif base_color == "green"
            "teal"
        elseif base_color == "orange"
            "navy"
        else
            "indigo"
        end
        
        # For single instance, we'll put info in title; for multiple, use shorter labels
        if length(files_data) == 1
            instance_label = ""  # Will be in main title
        else
            instance_label = "D$(info.dimension)_S$(info.seed)_$(info.data_type): "
        end
        
        # Plot 1: Bounds vs Time
        ax = axes[1, 1]
        ax.plot(df.time_seconds, df.lowerBound, "-", color=lower_color, alpha=0.8, linewidth=1.5, label="$(instance_label)Lower Bound")
        ax.plot(df.time_seconds, df.upperBound, "--", color=upper_color, alpha=0.8, linewidth=1.5, label="$(instance_label)Upper Bound")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Bound Value")
        ax.set_title("Bounds vs Time")
        ax.grid(true, alpha=0.3)
        
        # Plot 2: Bounds vs Iterations
        ax = axes[1, 2]
        ax.plot(df.iteration, df.lowerBound, "-", color=lower_color, alpha=0.8, linewidth=1.5, label="$(instance_label)Lower Bound")
        ax.plot(df.iteration, df.upperBound, "--", color=upper_color, alpha=0.8, linewidth=1.5, label="$(instance_label)Upper Bound")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Bound Value")
        ax.set_title("Bounds vs Iterations")
        ax.grid(true, alpha=0.3)
        
        # Plot 3: Gap vs Time
        ax = axes[2, 1]
        gap = df.upperBound .- df.lowerBound
        ax.plot(df.time_seconds, gap, "-", color=gap_color, alpha=0.8, linewidth=1.5, label="$(instance_label)Gap")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Gap (Upper - Lower)")
        ax.set_title("Optimality Gap vs Time")
        ax.set_yscale("log")
        ax.grid(true, alpha=0.3)
        
        # Plot 4: Gap vs Iterations
        ax = axes[2, 2]
        ax.plot(df.iteration, gap, "-", color=gap_color, alpha=0.8, linewidth=1.5, label="$(instance_label)Gap")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Gap (Upper - Lower)")
        ax.set_title("Optimality Gap vs Iterations")
        ax.set_yscale("log")
        ax.grid(true, alpha=0.3)
    end
    
    # Add legends only to the rightmost plots
    axes[1, 2].legend(bbox_to_anchor=(1.05, 1), loc="upper left")  # Upper right plot (bounds)
    axes[2, 2].legend(bbox_to_anchor=(1.05, 1), loc="upper left")  # Lower right plot (gaps)
    
    # Create title with instance information
    if length(files_data) == 1
        _, _, info = files_data[1]
        main_title = "Boscia Optimization Progress: Dimension $(info.dimension), Seed $(info.seed), Type $(info.data_type)"
    else
        main_title = "Boscia Optimization Progress: Multiple Instances"
    end
    
    plt.suptitle(main_title, fontsize=16, y=0.98)
    plt.tight_layout()
    
    if save_plots
        # Ensure plots directory exists
        plots_dir = "plots"
        if !isdir(plots_dir)
            mkdir(plots_dir)
        end
        
        # Generate filename based on plotted instances
        if length(files_data) == 1
            _, _, info = files_data[1]
            filename = "boscia_progress_D$(info.dimension)_S$(info.seed)_$(info.data_type).png"
        else
            filename = "boscia_progress_multiple_instances.png"
        end
        
        output_file = joinpath(plots_dir, filename)
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        println("Plot saved as: $output_file")
    end
    
    plt.show()
    
    return fig
end

function interactive_mode()
    """Interactive mode for selecting instances to plot."""
    
    println("=== Boscia Progress Plot Generator ===")
    println()
    
    # Show available instances
    println("Available instances:")
    all_files = find_matching_files(nothing, nothing, nothing)
    
    dimensions = Set()
    seeds = Set()
    data_types = Set()
    
    for (file, info) in all_files
        push!(dimensions, info.dimension)
        push!(seeds, info.seed)
        push!(data_types, info.data_type)
        println("  $(info.dimension) | $(info.seed) | $(info.data_type) | $(file)")
    end
    
    println()
    println("Available dimensions: $(sort(collect(dimensions)))")
    println("Available seeds: $(sort(collect(seeds)))")
    println("Available data types: $(sort(collect(data_types)))")
    println()
    
    # Get user input
    print("Enter dimension (or 'all'): ")
    dim_input = strip(readline())
    dimension = dim_input == "all" ? nothing : parse(Int, dim_input)
    
    print("Enter seed (or 'all'): ")
    seed_input = strip(readline())
    seed = seed_input == "all" ? nothing : parse(Int, seed_input)
    
    print("Enter data type ('cont', 'int', or 'all'): ")
    type_input = strip(readline())
    data_type = type_input == "all" ? nothing : type_input
    
    return dimension, seed, data_type
end

function main(args)
    """Main function to handle command line arguments or interactive mode."""
    
    dimension = nothing
    seed = nothing
    data_type = nothing
    
    if length(args) == 0
        # Interactive mode
        dimension, seed, data_type = interactive_mode()
    elseif length(args) == 3
        # Command line arguments
        dimension = parse(Int, args[1])
        seed = parse(Int, args[2])
        data_type = args[3]
    else
        println("Usage:")
        println("  julia plot_boscia_progress.jl                     # Interactive mode")
        println("  julia plot_boscia_progress.jl <dim> <seed> <type> # Direct specification")
        println("  Example: julia plot_boscia_progress.jl 100 1 cont")
        return
    end
    
    # Find matching files
    println("Searching for instances...")
    println("  Dimension: $(dimension === nothing ? "all" : dimension)")
    println("  Seed: $(seed === nothing ? "all" : seed)")
    println("  Data type: $(data_type === nothing ? "all" : data_type)")
    println()
    
    matching_files = find_matching_files(dimension, seed, data_type)
    
    if isempty(matching_files)
        println("No matching files found!")
        return
    end
    
    println("Found $(length(matching_files)) matching instance(s):")
    
    # Load data for all matching files
    files_data = []
    
    for (filename, info) in matching_files
        println("  Loading: $filename")
        try
            df = load_progress_data(filename)
            push!(files_data, (filename, df, info))
            println("    Loaded $(nrow(df)) progress points")
        catch e
            println("    Error loading $filename: $e")
        end
    end
    
    if isempty(files_data)
        println("No data could be loaded!")
        return
    end
    
    # Create plots
    println("\nGenerating plots...")
    plot_progress(files_data)
    
    println("Done!")
end

# Run the script
if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
