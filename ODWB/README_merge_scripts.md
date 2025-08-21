# CSV Merge Scripts for Solver Results

This directory contains scripts to merge CSV files from different solver setups, automatically separating continuous and integer examples.

## Main Script

### `merge_all_solver_results.jl`

The main script that processes all solver directories and creates merged files.

#### Usage Examples:

```bash
# Process all solver directories
julia merge_all_solver_results.jl

# Process specific solver(s)
julia merge_all_solver_results.jl Boscia
julia merge_all_solver_results.jl Boscia Pajarito
julia merge_all_solver_results.jl full_runs_boscia
```

## What the Script Does

1. **Finds solver directories** in `csv/` folder (e.g., Boscia, Pajarito, full_runs_boscia)

2. **Groups CSV files** by problem type:
   - **Continuous**: Files containing `_cont_`, `_continuous_`, `IND_0`, or `_0_`
   - **Integer**: Files containing `_int_`, `_integer_`, `IND_1`, or `_1_`

3. **Creates merged files** in each solver directory:
   - `boscia_continuous_merged.csv`
   - `boscia_integer_merged.csv`
   - `pajarito_continuous_merged.csv`
   - `pajarito_integer_merged.csv`
   - etc.

4. **Preserves original files** - individual CSV files remain untouched

5. **Validates results** and shows statistics

## File Structure

```
ODWB/
├── merge_all_solver_results.jl    # Main merging script
├── csv/
│   ├── Boscia/
│   │   ├── boscia_*.csv           # Individual result files
│   │   ├── boscia_continuous_merged.csv  # Generated merged file
│   │   └── boscia_integer_merged.csv     # Generated merged file
│   ├── Pajarito/
│   │   ├── pajarito_*.csv
│   │   └── pajarito_*_merged.csv
│   └── full_runs_boscia/
│       └── ...
```

## Output Format

Each merged file contains:
- **Header**: `seed;numberOfExperiments;numberOfParameters;N;time;solution;scaled_solution;dual_gap;rel_dual_gap;ncalls;num_nodes;termination;optimal_time;optimal_iteration`
- **Data rows**: All data from individual CSV files (excluding headers)

## Adding New Solvers

To add support for new solvers:

1. Create a directory under `csv/` with the solver name
2. Place CSV files in that directory
3. Ensure filenames contain problem type indicators:
   - For continuous: `_cont_`, `_continuous_`, `IND_0`, or `_0_`
   - For integer: `_int_`, `_integer_`, `IND_1`, or `_1_`
4. Run the script - it will automatically detect and process the new solver

## Example Output

```bash
$ julia merge_all_solver_results.jl

Merging solver results from: /path/to/ODWB/csv
Found solver directories: Boscia, Pajarito, full_runs_boscia

============================================================
Processing solver: Boscia
Directory: /path/to/ODWB/csv/Boscia
============================================================

Found 26 CSV files
  boscia_E_optimality_independent_cont__50_7_1.csv -> continuous
  boscia_E_optimality_independent_cont__50_7_2.csv -> continuous
  ...
  boscia_E_optimality_independent_int__50_7_3.csv -> integer

File groups identified:
  continuous: 25 files
  integer: 1 files

Processing Boscia - continuous (25 files)
  ✓ Created: boscia_continuous_merged.csv
    Total rows: 25
    Source files: 25/25
    Validated: 25 rows × 14 columns

Processing Boscia - integer (1 files)  
  ✓ Created: boscia_integer_merged.csv
    Total rows: 1
    Source files: 1/1
    Validated: 1 rows × 14 columns

============================================================
MERGE PROCESS COMPLETED
============================================================
Generated merged files:
  - csv/Boscia/boscia_continuous_merged.csv
  - csv/Boscia/boscia_integer_merged.csv
```
