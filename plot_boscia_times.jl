#!/usr/bin/env julia

"""
Plot script for Boscia runs showing average time and optimal_time per dimension
using geometric mean and variance.

Usage: 
    julia plot_boscia_times.jl                        # Plot baseline continuous data
    julia plot_boscia_times.jl heuristics             # Plot heuristics data (both continuous and integer)
    julia plot_boscia_times.jl heuristics continuous  # Plot heuristics continuous data only
    julia plot_boscia_times.jl baseline               # Plot baseline data
    julia plot_boscia_times.jl fedorov                # Plot fedorov data
    julia plot_boscia_times.jl follow_subgradient     # Plot follow_subgradient data
    julia plot_boscia_times.jl sr_rounding            # Plot sr_rounding data
    julia plot_boscia_times.jl pipage_rounding        # Plot pipage_rounding data
    julia plot_boscia_times.jl compare heuristics baseline  # Compare two sources
"""

using CSV, DataFrames, PyPlot, Statistics

function geometric_mean(x)
    """Calculate geometric mean of positive values, handling zeros and negative values."""
    # Filter out non-positive values for geometric mean calculation
    positive_x = filter(val -> val > 0, x)
    
    if isempty(positive_x)
        return 0.0
    end
    
    return exp(mean(log.(positive_x)))
end

function geometric_variance(x)
    """Calculate geometric variance of positive values."""
    # Filter out non-positive values
    positive_x = filter(val -> val > 0, x)
    
    if length(positive_x) <= 1
        return 0.0
    end
    
    log_x = log.(positive_x)
    return exp(var(log_x))
end

function get_data_paths(data_source::String)
    """Get the data file paths for a given data source."""
    
    if data_source == "heuristics"
        # Heuristics has both continuous and integer data
        return [
            "ODWB/csv/Boscia/heuristics/boscia_heuristics_continuous_merged.csv",
            "ODWB/csv/Boscia/heuristics/boscia_heuristics_integer_merged.csv"
        ]
    elseif data_source == "baseline"
        return [
            "ODWB/csv/Boscia/baseline/boscia_baseline_continuous_merged.csv",
            "ODWB/csv/Boscia/baseline/boscia_baseline_integer_merged.csv"
        ]
    elseif data_source == "fedorov"
        return [
            "ODWB/csv/Boscia/fedorov/boscia_fedorov_continuous_merged.csv",
            "ODWB/csv/Boscia/fedorov/boscia_fedorov_integer_merged.csv"
        ]
    elseif data_source == "follow_subgradient"
        return [
            "ODWB/csv/Boscia/follow_subgradient/boscia_follow_subgradient_continuous_merged.csv"
        ]
    elseif data_source == "sr_rounding"
        return [
            "ODWB/csv/Boscia/sr_rounding/boscia_sr_rounding_continuous_merged.csv",
            "ODWB/csv/Boscia/sr_rounding/boscia_sr_rounding_integer_merged.csv"
        ]
    elseif data_source == "pipage_rounding"
        return [
            "ODWB/csv/Boscia/pipage_rounding/boscia_pipage_rounding_continuous_merged.csv"
        ]
    else
        # Default/fallback paths for backward compatibility
        return [
            "ODWB/csv/Boscia/boscia_continuous_merged.csv",
            "ODWB/csv/full_runs_boscia/full_runs_boscia_continuous_merged.csv",
            "ODWB/csv/Boscia/full_runs_boscia_continuous_merged.csv"
        ]
    end
end

function load_data_files(data_source::String, data_types::Vector{String} = ["continuous", "integer"])
    """Load data files for the specified source and data types."""
    
    println("Loading Boscia $data_source data...")
    
    possible_paths = get_data_paths(data_source)
    data_files = Dict()
    
    for path in possible_paths
        if isfile(path)
            # Determine data type from filename
            data_type = if contains(path, "continuous")
                "continuous"
            elseif contains(path, "integer")
                "integer"
            else
                "continuous"  # default
            end
            
            # Only load if this data type is requested
            if data_type in data_types
                println("Found $data_type data file: $path")
                df = CSV.read(path, DataFrame, delim=';')
                println("Loaded $(nrow(df)) rows of $data_type data")
                data_files[data_type] = df
            else
                println("Skipping $data_type data file: $path")
            end
        end
    end
    
    if isempty(data_files)
        error("No data files found for source '$data_source' with types $(join(data_types, ", ")) in any of these locations: $(join(possible_paths, ", "))")
    end
    
    return data_files
end

function process_data_type(df, data_type_name)
    """Process data for a single type (continuous or integer)."""
    
    println("\nProcessing $data_type_name data...")
    
    # Filter out rows with missing numberOfExperiments
    valid_df = filter(row -> !ismissing(row.numberOfExperiments), df)
    println("Filtered $(nrow(df) - nrow(valid_df)) rows with missing numberOfExperiments")
    
    dimensions = sort(unique(valid_df.numberOfExperiments))
    println("Dimensions found: $dimensions")
    
    # Initialize arrays for results
    geom_means_time = Float64[]
    geom_vars_time = Float64[]
    geom_means_optimal = Float64[]
    geom_vars_optimal = Float64[]
    
    # Calculate geometric mean and variance for each dimension
    for dim in dimensions
        dim_data = filter(row -> row.numberOfExperiments == dim, valid_df)
        println("Processing dimension $dim: $(nrow(dim_data)) samples")
        
        # Extract time data
        times = dim_data.time
        optimal_times = dim_data.optimal_time ./ 1000.0  # Convert from milliseconds to seconds
        
        # Calculate geometric statistics for time
        geom_mean_t = geometric_mean(times)
        geom_var_t = geometric_variance(times)
        
        # Calculate geometric statistics for optimal_time
        geom_mean_opt = geometric_mean(optimal_times)
        geom_var_opt = geometric_variance(optimal_times)
        
        push!(geom_means_time, geom_mean_t)
        push!(geom_vars_time, geom_var_t)
        push!(geom_means_optimal, geom_mean_opt)
        push!(geom_vars_optimal, geom_var_opt)
        
        println("  Dimension $dim:")
        println("    Time - Geom Mean: $(round(geom_mean_t, digits=2)), Geom Var: $(round(geom_var_t, digits=2))")
        println("    Optimal Time - Geom Mean: $(round(geom_mean_opt, digits=2)), Geom Var: $(round(geom_var_opt, digits=2))")
    end
    
    return dimensions, geom_means_time, geom_vars_time, geom_means_optimal, geom_vars_optimal
end

function plot_boscia_times(data_source::String = "baseline", data_types::Vector{String} = ["continuous", "integer"])
    """Main plotting function for Boscia run times."""
    
    # Load data files
    data_files = load_data_files(data_source, data_types)
    
    # Set up the figure
    println("Creating plot...")
    figure(figsize=(15, 10))
    
    # Define colors and markers for different data types
    colors = Dict("continuous" => "blue", "integer" => "green")
    markers = Dict("continuous" => "o", "integer" => "s")
    
    plot_data = Dict()
    
    # Process each data type
    for (data_type, df) in data_files
        dimensions, geom_means_time, geom_vars_time, geom_means_optimal, geom_vars_optimal = process_data_type(df, data_type)
        
        # Store for plotting and summary
        plot_data[data_type] = (dimensions, geom_means_time, geom_vars_time, geom_means_optimal, geom_vars_optimal)
        
        # Calculate error bars (geometric standard deviation)
        geom_std_time = sqrt.(geom_vars_time)
        geom_std_optimal = sqrt.(geom_vars_optimal)
        
        # Plot algorithm time with error bars
        errorbar(dimensions, geom_means_time, yerr=geom_std_time, 
                 marker=markers[data_type], markersize=8, linewidth=2, capsize=5,
                 label="Algorithm Time ($data_type)", color=colors[data_type], alpha=0.8)
        
        # Plot optimal time with error bars (lighter version of same color)
        errorbar(dimensions, geom_means_optimal, yerr=geom_std_optimal,
                 marker=markers[data_type], markersize=6, linewidth=2, capsize=5, linestyle="--",
                 label="Optimal Time ($data_type)", color=colors[data_type], alpha=0.5)
    end
    
    # Formatting
    xlabel("Dimension (Number of Experiments)", fontsize=14)
    ylabel("Time (seconds) - Geometric Scale", fontsize=14)
    title("Boscia $(uppercasefirst(data_source)): Time Performance by Dimension\n(Geometric Mean ± Geometric Standard Deviation)", fontsize=16)
    
    # Set log scale for y-axis since we're dealing with geometric means
    yscale("log")
    
    # Grid and legend
    grid(true, alpha=0.3)
    legend(fontsize=12, loc="upper left")
    
    # Adjust layout
    tight_layout()
    
    # Save the plot
    plots_dir = "plots"
    if !isdir(plots_dir)
        mkdir(plots_dir)
    end
    
    # Create filename with data types suffix if not plotting all types
    if length(data_types) == 1
        filename = "boscia_$(data_source)_$(data_types[1])_times_by_dimension.png"
    else
        filename = "boscia_$(data_source)_times_by_dimension.png"
    end
    output_file = joinpath(plots_dir, filename)
    savefig(output_file, dpi=300, bbox_inches="tight")
    println("Plot saved as: $output_file")
    
    # Show summary statistics
    println("\nSummary Statistics:")
    println("=" ^ 50)
    
    for (data_type, (dimensions, geom_means_time, geom_vars_time, geom_means_optimal, geom_vars_optimal)) in plot_data
        println("\n$(uppercasefirst(data_type)) Data:")
        geom_std_time = sqrt.(geom_vars_time)
        geom_std_optimal = sqrt.(geom_vars_optimal)
        
        for (i, dim) in enumerate(dimensions)
            println("  Dimension $dim:")
            println("    Algorithm Time: $(round(geom_means_time[i], digits=2)) ± $(round(geom_std_time[i], digits=2))")
            println("    Optimal Time:   $(round(geom_means_optimal[i], digits=2)) ± $(round(geom_std_optimal[i], digits=2))")
        end
    end
    
    # Display the plot
    show()
    
    return plot_data
end

function compare_sources(sources::Vector{String}, data_types::Vector{String} = ["continuous"])
    """Compare optimal times between different data sources."""
    
    println("Comparing optimal times between sources: $(join(sources, " vs "))")
    
    # Set up the figure
    figure(figsize=(12, 8))
    
    # Define colors and markers for different sources
    colors = ["blue", "red", "green", "orange", "purple", "brown"]
    markers = ["o", "s", "^", "D", "v", "<"]
    
    all_plot_data = Dict()
    
    # Process each source
    for (idx, source) in enumerate(sources)
        println("\n" * "="^50)
        println("Loading source: $source")
        println("="^50)
        
        try
            # Load only continuous data for comparison
            data_files = load_data_files(source, data_types)
            
            # We'll focus on continuous data for comparison, but handle if only integer exists
            df = if haskey(data_files, "continuous")
                data_files["continuous"]
            elseif haskey(data_files, "integer") && "integer" in data_types
                data_files["integer"]
            else
                error("No suitable data found for source $source")
            end
            
            # Process the data
            actual_data_type = haskey(data_files, "continuous") ? "continuous" : "integer"
            dimensions, geom_means_time, geom_vars_time, geom_means_optimal, geom_vars_optimal = process_data_type(df, "$(source) $(actual_data_type)")
            
            # Store for summary
            all_plot_data[source] = (dimensions, geom_means_time, geom_vars_time, geom_means_optimal, geom_vars_optimal, actual_data_type)
            
            # Calculate error bars for optimal time
            geom_std_optimal = sqrt.(geom_vars_optimal)
            
            # Plot only optimal time for comparison
            color = colors[min(idx, length(colors))]
            marker = markers[min(idx, length(markers))]
            
            errorbar(dimensions, geom_means_optimal, yerr=geom_std_optimal,
                     marker=marker, markersize=8, linewidth=2, capsize=5,
                     label="$(uppercasefirst(source)) Optimal Time", color=color, alpha=0.8)
                     
        catch e
            println("Warning: Could not load data for source '$source': $e")
        end
    end
    
    # Formatting
    xlabel("Dimension (Number of Experiments)", fontsize=14)
    ylabel("Optimal Time (seconds) - Geometric Scale", fontsize=14)
    title("Optimal Time Comparison: $(join([uppercasefirst(s) for s in sources], " vs "))\n(Geometric Mean ± Geometric Standard Deviation)", fontsize=16)
    
    # Set log scale for y-axis
    yscale("log")
    
    # Grid and legend
    grid(true, alpha=0.3)
    legend(fontsize=12, loc="upper left")
    
    # Adjust layout
    tight_layout()
    
    # Save the plot
    plots_dir = "plots"
    if !isdir(plots_dir)
        mkdir(plots_dir)
    end
    
    filename = "boscia_optimal_time_comparison_$(join(sources, "_vs_")).png"
    output_file = joinpath(plots_dir, filename)
    savefig(output_file, dpi=300, bbox_inches="tight")
    println("Comparison plot saved as: $output_file")
    
    # Show summary statistics
    println("\nComparison Summary:")
    println("=" ^ 60)
    
    for (source, (dimensions, geom_means_time, geom_vars_time, geom_means_optimal, geom_vars_optimal, data_type)) in all_plot_data
        println("\n$(uppercasefirst(source)) ($data_type):")
        geom_std_optimal = sqrt.(geom_vars_optimal)
        
        for (i, dim) in enumerate(dimensions)
            println("  Dimension $dim: $(round(geom_means_optimal[i], digits=2)) ± $(round(geom_std_optimal[i], digits=2)) seconds")
        end
    end
    
    # Display the plot
    show()
    
    return all_plot_data
end

# Run the main function if script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    try
        if length(ARGS) > 0 && ARGS[1] == "compare"
            # Comparison mode
            if length(ARGS) < 3
                error("Comparison mode requires at least 2 sources. Usage: julia plot_boscia_times.jl compare source1 source2 [source3...]")
            end
            
            sources = ARGS[2:end]
            println("Comparison mode: comparing sources $(join(sources, ", "))")
            results = compare_sources(sources, ["continuous"])
            println("Comparison plot generation completed successfully!")
        else
            # Single source mode
            data_source = length(ARGS) > 0 ? ARGS[1] : "baseline"
            
            # Check if specific data type is requested
            data_types = if length(ARGS) > 1 && ARGS[2] == "continuous"
                ["continuous"]
            elseif length(ARGS) > 1 && ARGS[2] == "integer"
                ["integer"]
            else
                ["continuous", "integer"]
            end
            
            println("Plotting data for source: $data_source, types: $(join(data_types, ", "))")
            results = plot_boscia_times(data_source, data_types)
            println("Plot generation completed successfully!")
        end
    catch e
        println("Error: $e")
        println("Stacktrace:")
        for (exc, bt) in Base.catch_stack()
            showerror(stdout, exc, bt)
            println()
        end
        exit(1)
    end
end
