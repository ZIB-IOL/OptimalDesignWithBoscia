#!/usr/bin/env julia

"""
Plot script for Boscia continuous runs showing average time and optimal_time per dimension
using geometric mean and variance.

Usage: julia plot_boscia_times.jl
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

function plot_boscia_times()
    """Main plotting function for Boscia continuous run times."""
    
    # Load data
    println("Loading Boscia continuous data...")
    data_path = "ODWB/csv/Boscia/boscia_continuous_merged.csv"
    
    if !isfile(data_path)
        error("Data file not found: $data_path")
    end
    
    df = CSV.read(data_path, DataFrame, delim=';')
    println("Loaded $(nrow(df)) rows of data")
    
    # Get unique dimensions (numberOfExperiments)
    dimensions = sort(unique(df.numberOfExperiments))
    println("Dimensions found: $dimensions")
    
    # Initialize arrays for results
    geom_means_time = Float64[]
    geom_vars_time = Float64[]
    geom_means_optimal = Float64[]
    geom_vars_optimal = Float64[]
    
    # Calculate geometric mean and variance for each dimension
    for dim in dimensions
        dim_data = filter(row -> row.numberOfExperiments == dim, df)
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
    
    # Create the plot
    println("Creating plot...")
    
    # Set up the figure
    figure(figsize=(12, 8))
    
    # Calculate error bars (geometric standard deviation)
    geom_std_time = sqrt.(geom_vars_time)
    geom_std_optimal = sqrt.(geom_vars_optimal)
    
    # Plot with error bars
    errorbar(dimensions, geom_means_time, yerr=geom_std_time, 
             marker="o", markersize=8, linewidth=2, capsize=5,
             label="Algorithm Time", color="blue", alpha=0.8)
    
    errorbar(dimensions, geom_means_optimal, yerr=geom_std_optimal,
             marker="s", markersize=8, linewidth=2, capsize=5,
             label="Optimal Time", color="red", alpha=0.8)
    
    # Formatting
    xlabel("Dimension (Number of Experiments)", fontsize=14)
    ylabel("Time (seconds) - Geometric Scale", fontsize=14)
    title("Boscia Continuous Runs: Time Performance by Dimension\n(Geometric Mean ± Geometric Standard Deviation)", fontsize=16)
    
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
    
    filename = "boscia_times_by_dimension.png"
    output_file = joinpath(plots_dir, filename)
    savefig(output_file, dpi=300, bbox_inches="tight")
    println("Plot saved as: $output_file")
    
    # Show summary statistics
    println("\nSummary Statistics:")
    println("=" ^ 50)
    for (i, dim) in enumerate(dimensions)
        println("Dimension $dim:")
        println("  Algorithm Time: $(round(geom_means_time[i], digits=2)) ± $(round(geom_std_time[i], digits=2))")
        println("  Optimal Time:   $(round(geom_means_optimal[i], digits=2)) ± $(round(geom_std_optimal[i], digits=2))")
        println()
    end
    
    # Display the plot
    show()
    
    return dimensions, geom_means_time, geom_vars_time, geom_means_optimal, geom_vars_optimal
end

# Run the main function if script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    try
        results = plot_boscia_times()
        println("Plot generation completed successfully!")
    catch e
        println("Error: $e")
        exit(1)
    end
end
