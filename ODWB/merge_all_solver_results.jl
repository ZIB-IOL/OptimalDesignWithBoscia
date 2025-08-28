#!/usr/bin/env julia

"""
Central script to merge CSV files from all solver setups.
Usage: 
  julia merge_all_solver_results.jl                    # Process all solver directories
  julia merge_all_solver_results.jl Boscia             # Process specific solver
  julia merge_all_solver_results.jl Boscia Pajarito    # Process multiple specific solvers

This script will:
- Find all solver directories in csv/
- Group CSV files by solver and problem type (continuous/integer)
- Create separate merged files for each combination
- Place merged files in each solver's directory
- Preserve individual files
"""

using CSV, DataFrames

function identify_file_type(filename, solver_dir)
    """Identify solver, problem type, and other metadata from filename and directory."""
    
    # Remove .csv extension
    base_name = replace(filename, ".csv" => "")
    
    # Use directory name as primary solver identifier
    solver = lowercase(solver_dir)
    problem_type = "unknown"
    
    # Determine problem type from filename patterns
    # New pattern: boscia_E_optimality_independent_cont__ and boscia_E_optimality_independent_int__
    if contains(base_name, "_cont__") || contains(base_name, "_cont_") || contains(base_name, "_continuous_")
        problem_type = "continuous"
    elseif contains(base_name, "_int__") || contains(base_name, "_int_") || contains(base_name, "_integer_")
        problem_type = "integer"
    else
        # Try to infer from other patterns
        if contains(base_name, "IND_0") || contains(base_name, "_0_") || contains(base_name, "continuous")
            problem_type = "continuous"
        elseif contains(base_name, "IND_1") || contains(base_name, "_1_") || contains(base_name, "integer")
            problem_type = "integer"
        end
    end
    
    return (solver = solver, problem_type = problem_type, filename = filename)
end

function process_solver_directory(solver_dir, csv_base_path)
    """Process all CSV files in a specific solver directory, including subdirectories."""
    
    solver_path = joinpath(csv_base_path, solver_dir)
    
    if !isdir(solver_path)
        println("Warning: Directory $solver_path does not exist")
        return false
    end
    
    println("\n" * "="^60)
    println("Processing solver: $solver_dir")
    println("Directory: $solver_path")
    println("="^60)
    
    # Find all CSV files recursively (excluding previously merged ones)
    csv_files = String[]
    csv_paths = String[]
    
    function collect_csv_files(dir, relative_prefix = "")
        items = readdir(dir)
        for item in items
            full_path = joinpath(dir, item)
            relative_path = isempty(relative_prefix) ? item : joinpath(relative_prefix, item)
            
            if isdir(full_path)
                # Recursively search subdirectories
                collect_csv_files(full_path, relative_path)
            elseif endswith(item, ".csv") && !contains(item, "_merged")
                push!(csv_files, item)
                push!(csv_paths, full_path)
                println("  Found: $relative_path")
            end
        end
    end
    
    collect_csv_files(solver_path)
    
    if isempty(csv_files)
        println("No CSV files found in $solver_dir (including subdirectories)!")
        return false
    end
    
    println("Found $(length(csv_files)) CSV files total")
    
    # Group files by problem type
    file_groups = Dict()
    
    for (i, file) in enumerate(csv_files)
        file_info = identify_file_type(file, solver_dir)
        
        # Skip files with unknown problem type
        if file_info.problem_type == "unknown"
            println("Skipping unknown problem type: $file")
            continue
        end
        
        key = file_info.problem_type
        
        if !haskey(file_groups, key)
            file_groups[key] = []
        end
        
        # Store both filename and full path
        push!(file_groups[key], (filename = file, path = csv_paths[i]))
        println("  $file -> $(file_info.problem_type)")
    end
    
    if isempty(file_groups)
        println("No valid files found to merge in $solver_dir!")
        return false
    end
    
    println("\nFile groups identified:")
    for (group_name, files) in file_groups
        println("  $group_name: $(length(files)) files")
    end
    
    # Process each group
    successful_merges = 0
    
    for (problem_type, files) in file_groups
        println("\nProcessing $solver_dir - $problem_type ($(length(files)) files)")
        
        output_file = "$(lowercase(solver_dir))_$(problem_type)_merged.csv"
        output_path = joinpath(solver_path, output_file)
        
        merged_data = String[]
        total_rows = 0
        processed_files = 0
        detected_header = nothing
        
        for file_info in sort(files, by = x -> x.filename)  # Sort for consistent ordering
            println("  Processing: $(file_info.filename)")
            
            try
                lines = readlines(file_info.path)
                
                # Remove empty lines
                lines = filter(line -> !isempty(strip(line)), lines)
                
                if isempty(lines)
                    println("    Empty file, skipping...")
                    continue
                end
                
                # Detect header from first file and skip headers from all files
                data_lines = String[]
                for (i, line) in enumerate(lines)
                    clean_line = strip(line)
                    
                    # Detect header from first non-empty line that starts with "seed"
                    if detected_header === nothing && (startswith(clean_line, "seed;") || startswith(clean_line, "seed,"))
                        detected_header = clean_line
                        println("    Detected header format: $detected_header")
                        continue  # Skip this header line
                    end
                    
                    # Skip any header lines and empty lines
                    if !startswith(clean_line, "seed;") && !startswith(clean_line, "seed,") && !isempty(clean_line)
                        # Additional validation: check if the line has the expected number of semicolons/commas
                        delimiter = contains(detected_header === nothing ? ";" : detected_header, ";") ? ';' : ','
                        expected_fields = length(split(detected_header === nothing ? "seed;col1;col2" : detected_header, delimiter))
                        actual_fields = length(split(clean_line, delimiter))
                        
                        # Only include lines that have a reasonable number of fields (at least 3, and not too far from expected)
                        if actual_fields >= 3 && abs(actual_fields - expected_fields) <= 1
                            push!(data_lines, clean_line)
                        else
                            println("    Skipping malformed line with $actual_fields fields (expected ~$expected_fields): $(length(clean_line) > 50 ? clean_line[1:50] * "..." : clean_line)")
                        end
                    end
                end
                
                append!(merged_data, data_lines)
                println("    Added $(length(data_lines)) rows")
                total_rows += length(data_lines)
                processed_files += 1
                
            catch e
                println("    Error reading $(file_info.filename): $e")
                continue
            end
        end
        
        # Write merged file
        if !isempty(merged_data) && detected_header !== nothing
            try
                # Deduplicate data while preserving order
                unique_data = String[]
                seen_lines = Set{String}()
                
                for line in merged_data
                    if !(line in seen_lines)
                        push!(unique_data, line)
                        push!(seen_lines, line)
                    end
                end
                
                if length(unique_data) != length(merged_data)
                    println("    Removed $(length(merged_data) - length(unique_data)) duplicate lines")
                end
                
                open(output_path, "w") do f
                    # Write detected header
                    println(f, detected_header)
                    
                    # Write deduplicated data
                    for line in unique_data
                        println(f, line)
                    end
                end
                
                println("\n  ✓ Created: $output_file")
                println("    Total rows: $(length(unique_data)) (from $total_rows raw rows)")
                println("    Source files: $processed_files/$(length(files))")
                
                # Validate the merged file
                try
                    # Use appropriate delimiter based on detected header
                    delimiter = contains(detected_header, ";") ? ';' : ','
                    df = CSV.read(output_path, DataFrame, delim=delimiter)
                    println("    Validated: $(nrow(df)) rows × $(ncol(df)) columns")
                    
                    # Show some basic statistics
                    if "numberOfExperiments" in names(df) && nrow(df) > 0
                        unique_experiments = sort(unique(df.numberOfExperiments))
                        println("    Experiment sizes: $unique_experiments")
                    end
                    
                    if "termination" in names(df) && nrow(df) > 0
                        termination_counts = combine(groupby(df, :termination), nrow => :count)
                        println("    Termination status:")
                        for row in eachrow(termination_counts)
                            println("      $(row.termination): $(row.count)")
                        end
                    end
                    
                catch e
                    println("    Warning: Could not validate as DataFrame: $e")
                end
                
                successful_merges += 1
                
            catch e
                println("  ✗ Error writing $output_file: $e")
            end
        elseif isempty(merged_data)
            println("  ✗ No data found for $solver_dir - $problem_type")
        else
            println("  ✗ No header detected for $solver_dir - $problem_type")
        end
    end
    
    println("\n$solver_dir summary: $successful_merges/$(length(file_groups)) groups processed successfully")
    return successful_merges > 0
end

function merge_all_solvers(target_solvers = nothing)
    """Main function to process all or specified solver directories."""
    
    # Base paths
    base_dir = pwd()
    csv_base_path = joinpath(base_dir, "csv")
    
    println("Merging solver results from: $csv_base_path")
    
    if !isdir(csv_base_path)
        println("Error: csv directory not found at $csv_base_path")
        println("Make sure you're running this script from the ODWB directory")
        return
    end
    
    # Find all solver directories (exclude full_runs directories)
    all_dirs = filter(d -> isdir(joinpath(csv_base_path, d)) && !startswith(d, "full_runs"), readdir(csv_base_path))
    
    if target_solvers === nothing
        # Process all directories
        solver_dirs = all_dirs
        println("Found solver directories: $(join(solver_dirs, ", "))")
    else
        # Process only specified solvers
        solver_dirs = filter(d -> d in target_solvers, all_dirs)
        missing_solvers = filter(s -> !(s in all_dirs), target_solvers)
        
        if !isempty(missing_solvers)
            println("Warning: The following solvers were not found: $(join(missing_solvers, ", "))")
        end
        
        if isempty(solver_dirs)
            println("Error: No valid solver directories found")
            return
        end
        
        println("Processing specified solvers: $(join(solver_dirs, ", "))")
    end
    
    # Process each solver directory
    total_success = 0
    total_attempted = 0
    
    for solver_dir in solver_dirs
        total_attempted += 1
        if process_solver_directory(solver_dir, csv_base_path)
            total_success += 1
        end
    end
    
    # Final summary
    println("\n" * "="^60)
    println("MERGE PROCESS COMPLETED")
    println("="^60)
    println("Successfully processed: $total_success/$total_attempted solver directories")
    
    # List all generated merged files
    println("\nGenerated merged files:")
    for solver_dir in solver_dirs
        solver_path = joinpath(csv_base_path, solver_dir)
        if isdir(solver_path)
            merged_files = filter(f -> contains(f, "_merged.csv"), readdir(solver_path))
            for file in merged_files
                rel_path = joinpath("csv", solver_dir, file)
                println("  - $rel_path")
            end
        end
    end
    
    if total_success == 0
        println("\nNo files were successfully merged. Check the file formats and directory structure.")
    else
        println("\nMerge completed successfully!")
    end
end

# Main execution
if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) == 0
        # No arguments: process all solver directories
        merge_all_solvers()
    else
        # Arguments provided: process specific solvers
        target_solvers = ARGS
        merge_all_solvers(target_solvers)
    end
end
