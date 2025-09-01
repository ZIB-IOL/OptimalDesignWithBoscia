#!/usr/bin/env julia

"""
Script to compare heuristic performance for Boscia solver results.
This script analyzes CSV files from the main Boscia directory (baseline) and 
subdirectories (different heuristics) to compute:
- Geometric mean of optimal_time
- Arithmetic mean of optimal_iteration
- Geometric mean of dual_gap
- Standard deviations for all metrics

Usage:
  julia compare_heuristics.jl                    # Compare all available heuristics
  julia compare_heuristics.jl fedorov            # Compare specific heuristic only
  julia compare_heuristics.jl fedorov sr_rounding # Compare multiple specific heuristics
  julia compare_heuristics.jl --latex            # Generate only LaTeX code for all heuristics
  julia compare_heuristics.jl fedorov --latex    # Generate only LaTeX code for specific heuristics

Output: Formatted table with comparison metrics + LaTeX table code
LaTeX flag (--latex or -l): Generate only LaTeX code for easy copy-pasting
"""

using CSV, DataFrames, Statistics, Printf

function geometric_mean(values)
    """Compute geometric mean of positive values, handling zeros by adding small epsilon."""
    # Filter out non-positive values and convert to Float64
    positive_values = Float64[]
    for v in values
        if !ismissing(v) && !isnan(v) && v > 0
            push!(positive_values, Float64(v))
        end
    end
    
    if isempty(positive_values)
        return NaN
    end
    
    # Compute geometric mean: exp(mean(log(values)))
    return exp(mean(log.(positive_values)))
end

function geometric_std(values)
    """Compute geometric standard deviation."""
    # Filter out non-positive values and convert to Float64
    positive_values = Float64[]
    for v in values
        if !ismissing(v) && !isnan(v) && v > 0
            push!(positive_values, Float64(v))
        end
    end
    
    if length(positive_values) < 2
        return NaN
    end
    
    # Geometric std: exp(std(log(values)))
    log_values = log.(positive_values)
    return exp(std(log_values))
end

function load_heuristic_data(csv_path, heuristic_name)
    """Load data from a CSV file and return DataFrame with heuristic identifier."""
    try
        df = CSV.read(csv_path, DataFrame, delim=';')
        df.heuristic = fill(heuristic_name, nrow(df))
        return df
    catch e
        println("Warning: Could not load $csv_path: $e")
        return DataFrame()
    end
end

function find_boscia_files(base_path)
    """Find all Boscia merged CSV files (main directory and subdirectories)."""
    files_info = []
    
    boscia_path = joinpath(base_path, "Boscia")
    if !isdir(boscia_path)
        error("Boscia directory not found at $boscia_path")
    end
    
    # Main directory files (baseline)
    for file in readdir(boscia_path)
        if endswith(file, "_continuous_merged.csv") && startswith(file, "boscia_")
            file_path = joinpath(boscia_path, file)
            if isfile(file_path)
                push!(files_info, (path=file_path, heuristic="baseline", type="continuous"))
            end
        elseif endswith(file, "_integer_merged.csv") && startswith(file, "boscia_")
            file_path = joinpath(boscia_path, file)
            if isfile(file_path)
                push!(files_info, (path=file_path, heuristic="baseline", type="integer"))
            end
        end
    end
    
    # Subdirectory files (heuristics)
    subdirs = filter(d -> isdir(joinpath(boscia_path, d)) && !contains(d, "_merged"), readdir(boscia_path))
    
    for subdir in subdirs
        subdir_path = joinpath(boscia_path, subdir)
        for file in readdir(subdir_path)
            if endswith(file, "_continuous_merged.csv")
                file_path = joinpath(subdir_path, file)
                if isfile(file_path)
                    push!(files_info, (path=file_path, heuristic=subdir, type="continuous"))
                end
            elseif endswith(file, "_integer_merged.csv")
                file_path = joinpath(subdir_path, file)
                if isfile(file_path)
                    push!(files_info, (path=file_path, heuristic=subdir, type="integer"))
                end
            end
        end
    end
    
    return files_info
end

function compute_metrics(df)
    """Compute all required metrics for a dataset."""
    if nrow(df) == 0
        return (
            optimal_time_geomean = NaN,
            optimal_time_geostd = NaN,
            optimal_iteration_mean = NaN,
            optimal_iteration_std = NaN,
            dual_gap_geomean = NaN,
            dual_gap_geostd = NaN,
            n_samples = 0
        )
    end
    
    # Extract columns, handling potential missing values
    # Convert optimal_time from milliseconds to seconds
    optimal_times_ms = skipmissing(df.optimal_time)
    optimal_times_sec = [t / 1000.0 for t in optimal_times_ms]  # Convert ms to seconds
    optimal_iterations = skipmissing(df.optimal_iteration)
    dual_gaps = skipmissing(df.dual_gap)
    
    return (
        optimal_time_geomean = geometric_mean(optimal_times_sec),
        optimal_time_geostd = geometric_std(optimal_times_sec),
        optimal_iteration_mean = mean(optimal_iterations),
        optimal_iteration_std = std(optimal_iterations),
        dual_gap_geomean = geometric_mean(dual_gaps),
        dual_gap_geostd = geometric_std(dual_gaps),
        n_samples = nrow(df)
    )
end

function format_metric(value, std_value, is_geometric=false)
    """Format a metric with its standard deviation."""
    if isnan(value)
        return "N/A"
    end
    
    if isnan(std_value)
        return @sprintf("%.3e", value)
    end
    
    return @sprintf("%.3e ± %.3e", value, std_value)
end

function format_metric_latex(value, std_value, is_geometric=false)
    """Format a metric with its standard deviation for LaTeX."""
    if isnan(value)
        return "N/A"
    end
    
    if isnan(std_value)
        return @sprintf("%.3e", value)
    end
    
    # Use \pm for the ± symbol in LaTeX
    return @sprintf("%.3e \$\\pm\$ %.3e", value, std_value)
end

function generate_latex_table(results, problem_type)
    """Generate LaTeX table code for the results."""
    println("\n% LaTeX table for $(uppercase(problem_type)) problems")
    println("\\begin{table}[htbp]")
    println("\\centering")
    println("\\caption{Heuristic Performance Comparison - $(titlecase(problem_type)) Problems}")
    println("\\label{tab:heuristics_$(problem_type)}")
    println("\\begin{tabular}{lcccc}")
    println("\\toprule")
    println("Heuristic & N Samples & Optimal Time (s) & Optimal Iteration & Dual Gap \\\\")
    println("& & (Geometric Mean) & (Arithmetic Mean) & (Geometric Mean) \\\\")
    println("\\midrule")
    
    for result in results
        heuristic_display = result.heuristic == "baseline" ? "Baseline" : titlecase(replace(result.heuristic, "_" => " "))
        
        optimal_time_str = format_metric_latex(result.optimal_time_geomean, result.optimal_time_geostd, true)
        optimal_iter_str = format_metric_latex(result.optimal_iteration_mean, result.optimal_iteration_std, false)
        dual_gap_str = format_metric_latex(result.dual_gap_geomean, result.dual_gap_geostd, true)
        
        println("$(heuristic_display) & $(result.n_samples) & $(optimal_time_str) & $(optimal_iter_str) & $(dual_gap_str) \\\\")
    end
    
    println("\\bottomrule")
    println("\\end{tabular}")
    println("\\begin{tablenotes}")
    println("\\small")
    println("\\item Optimal time converted from milliseconds to seconds.")
    println("\\item Geometric mean used for optimal time and dual gap.")
    println("\\item Arithmetic mean used for optimal iteration.")
    println("\\item \$\\pm\$ indicates standard deviation (geometric for geometric means).")
    println("\\end{tablenotes}")
    println("\\end{table}")
end

function compare_heuristics(target_heuristics=nothing, latex_only=false)
    """Main function to compare heuristics performance."""
    
    # Base paths
    base_dir = pwd()
    csv_base_path = joinpath(base_dir, "csv")
    
    if !latex_only
        println("Comparing Boscia heuristics performance...")
        println("Base directory: $csv_base_path")
    end
    
    if !isdir(csv_base_path)
        error("CSV directory not found at $csv_base_path")
    end
    
    # Find all Boscia files
    files_info = find_boscia_files(csv_base_path)
    
    if isempty(files_info)
        error("No Boscia merged CSV files found!")
    end
    
    # Filter by target heuristics if specified
    if target_heuristics !== nothing
        target_set = Set(["baseline"; target_heuristics])
        files_info = filter(f -> f.heuristic in target_set, files_info)
    end
    
    if !latex_only
        println("Found files:")
        for info in files_info
            println("  $(info.heuristic) ($(info.type)): $(info.path)")
        end
    end
    
    # Load and combine data
    all_data = DataFrame()
    
    for info in files_info
        df = load_heuristic_data(info.path, info.heuristic)
        if nrow(df) > 0
            df.problem_type = fill(info.type, nrow(df))
            all_data = isempty(all_data) ? df : vcat(all_data, df, cols=:union)
        end
    end
    
    if nrow(all_data) == 0
        error("No data could be loaded from the CSV files!")
    end
    
    if !latex_only
        println("\nLoaded $(nrow(all_data)) total records")
    end
    
    # Process by problem type
    for problem_type in ["continuous", "integer"]
        type_data = filter(row -> row.problem_type == problem_type, all_data)
        
        if nrow(type_data) == 0
            if !latex_only
                println("\nNo $problem_type data found, skipping...")
            end
            continue
        end
        
        if !latex_only
            println("\n" * "="^80)
            println("COMPARISON RESULTS - $(uppercase(problem_type)) PROBLEMS")
            println("="^80)
        end
        
        # Group by heuristic and compute metrics
        heuristic_groups = groupby(type_data, :heuristic)
        results = []
        
        for group in heuristic_groups
            heuristic_name = group.heuristic[1]
            metrics = compute_metrics(group)
            push!(results, (heuristic=heuristic_name, metrics...))
        end
        
        # Sort results with baseline first
        sort!(results, by = x -> x.heuristic == "baseline" ? "0_baseline" : x.heuristic)
        
        if !latex_only
            # Print header
            println()
            println(@sprintf("%-20s %12s %20s %20s %20s", 
                    "Heuristic", "N", "Optimal Time (s)", "Optimal Iteration", "Dual Gap"))
            println(@sprintf("%-20s %12s %20s %20s %20s", 
                    "", "Samples", "(Geometric Mean)", "(Arithmetic Mean)", "(Geometric Mean)"))
            println("-"^93)
            
            # Print results
            for result in results
                heuristic_display = result.heuristic == "baseline" ? "baseline*" : result.heuristic
                
                optimal_time_str = format_metric(result.optimal_time_geomean, result.optimal_time_geostd, true)
                optimal_iter_str = format_metric(result.optimal_iteration_mean, result.optimal_iteration_std, false)
                dual_gap_str = format_metric(result.dual_gap_geomean, result.dual_gap_geostd, true)
                
                println(@sprintf("%-20s %12d %20s %20s %20s", 
                        heuristic_display, 
                        result.n_samples,
                        optimal_time_str,
                        optimal_iter_str,
                        dual_gap_str))
            end
            
            println()
            println("* baseline: results from main Boscia directory")
            println("Optimal time converted from milliseconds to seconds")
            println("Geometric mean used for optimal_time and dual_gap")
            println("Arithmetic mean used for optimal_iteration")
            println("± indicates standard deviation (geometric for geometric means)")
        end
        
        # Generate LaTeX table
        generate_latex_table(results, problem_type)
    end
    
    if !latex_only
        println("\n" * "="^80)
        println("COMPARISON COMPLETED")
        println("="^80)
    end
end

# Main execution
if abspath(PROGRAM_FILE) == @__FILE__
    # Check for latex-only flag
    latex_only = "--latex" in ARGS || "-l" in ARGS
    
    # Filter out the latex flag from heuristic arguments
    heuristic_args = filter(arg -> arg != "--latex" && arg != "-l", ARGS)
    
    if length(heuristic_args) == 0
        # No heuristic arguments: compare all heuristics
        compare_heuristics(nothing, latex_only)
    else
        # Arguments provided: compare specific heuristics
        compare_heuristics(heuristic_args, latex_only)
    end
end
