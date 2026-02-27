## CSV merge, aggregation, and LaTeX scripts

This README documents the three Julia scripts that handle:

- merging single-run CSVs into merged CSVs,
- aggregating merged results into summary CSVs, and
- generating publication-ready LaTeX tables.

These scripts live in `ODWB/`:

- `merge_single_runs_to_csv.jl`
- `aggregate_merged_by_group.jl`
- `aggregated_to_tex.jl`

and operate on CSVs under `ODWB/csv/`.

The scripts support:

- **E-optimal design** (EOPT)
- **Boscia smoothing experiments** (4 regimes)
- **Algebraic Graph Connectivity** (AGC) – connected / disconnected
  
for both **Boscia** and **SCIP-SDP** (SCIPSDP oa/bnb), and for **independent** vs **correlated** data where appropriate.

---

## 1. `merge_single_runs_to_csv.jl`

### Purpose

Merge many *single-run* CSVs (one line per optimization run) into **merged CSVs** with:

- a fixed **grid of expected instances** per group,
- **placeholder rows** for missing runs (time limit, `Inf` solution, stats = 0),
- clear **printout of missing instances**.

This script **never modifies** the original single-run CSVs.

### Instance grids

All grids are defined in the script; they are consistent with how instances are generated in `run_optimal_design.jl`.

- **E-optimal design (EOPT, Boscia + SCIPSDP):**
  - Dimensions \(m\): `50, 80, 100, 120, 150`
  - Seeds: `1:5`
  - For each \(m\), 3 values of \(N\) (rank_deficient, one, log), inferred from `n = floor(sqrt(m))`.
  - → **75 expected instances** per group (5 \(m\) × 5 seeds × 3 \(N\)).

- **Boscia smoothing (EOPT, 4 regimes):**
  - Dimensions \(m\): `50, 100, 150, 200`
  - Seeds: `1:3`
  - Same 3 \(N\) constructions per \(m\) (rank_deficient, one, log).
  - → **36 expected instances** per regime & data type (4 \(m\) × 3 seeds × 3 \(N\)).

- **AGC (Algebraic Graph Connectivity, Boscia + SCIPSDP):**
  - Dimensions \(m\): `50, 80, 100, 150, 200`
  - Seeds: `1:5`
  - For each \(m\), exactly one \((n, N)\) pair (e.g. 50→(16,25), 80→(26,40), …), inferred by scanning filenames.
  - → **25 expected instances** per group (5 \(m\) × 5 seeds).

### Placeholders for missing runs

For any expected instance without a single-run CSV, the script inserts a placeholder row with:

- `time = 3600` (TIME_LIMIT),
- `solution = Inf`, `scaled_solution = Inf`,
- `termination = "ERROR"`,
- stats (`ncalls`, `num_nodes`, `n_cuts_*`, `n_sdp_iters`, …) set to `0`,
- and for Boscia a `solution_source = "missing"` flag.

The script prints, per group, something like:

```text
--- Boscia smoothing large_mu correlated ---
Found 9/36 runs (27 missing)
Missing instances:
  m=100 n=10 N=7 seed=1
  ...
```

### Filenames and groups

#### E-optimal design (EOPT)

- **Boscia single-run CSVs** (semicolon-delimited):

  ```text
  csv/Boscia/boscia__E_optimality_{independent|correlated}_cont__m_n_N_seed.csv
  ```

- **SCIP-SDP single-run CSVs** (comma-delimited):

  ```text
  csv/SCIPSDP/scip_sdp_oa_E_optimality_{independent|correlated}_cont__m_n_N_seed.csv
  csv/SCIPSDP/scip_sdp_bnb_E_optimality_{independent|correlated}_cont__m_n_N_seed.csv
  ```

- **Merged CSVs**:

  ```text
  csv/Boscia/boscia_E_optimality_{independent|correlated}_cont_merged.csv
  csv/SCIPSDP/scip_sdp_{oa|bnb}_E_optimality_{independent|correlated}_cont_merged.csv
  ```

Each merged file has exactly **75 rows**, one per expected (m,n,N,seed) quadruple.

#### Boscia smoothing (EOPT)

Four regimes are inferred from the filename prefix `boscia_<mu0>_<decay>_<mu_min>_E_optimality_…`:

- `large_mu`: decay ≈ 1.0, μ_min < 0.001  
- `small_mu`: decay ≈ 1.0, μ_min ≥ 0.001  
- `decay_0.9`: decay ≈ 0.9  
- `decay_0.7`: decay ≈ 0.7

Single-run filenames (semicolon-delimited) are:

```text
csv/Boscia/boscia_<mu0>_<decay>_<mu_min>_E_optimality_{independent|correlated}__m_n_N_seed.csv
```

Merged smoothing files:

```text
csv/Boscia/boscia_smoothing_{large_mu|small_mu|decay_0.9|decay_0.7}_E_optimality_{independent|correlated}_merged.csv
```

Each merged smoothing file has exactly **36 rows** (4 m × 3 seeds × 3 N).

#### AGC (Algebraic Graph Connectivity)

Single-run filenames:

- **Boscia** (semicolon-delimited):

  ```text
  csv/Boscia/boscia__AGC_optimality_{independent|correlated}_{connected|disconnected}_m_n_N_seed.csv
  ```

- **SCIP-SDP** (comma-delimited):

  ```text
  csv/SCIPSDP/scip_sdp_{oa|bnb}_AGC_optimality_{independent|correlated}_{connected|disconnected}_m_n_N_seed.csv
  ```

Merged AGC files:

```text
csv/Boscia/boscia_AGC_optimality_{independent|correlated}_{connected|disconnected}_merged.csv
csv/SCIPSDP/scip_sdp_{oa|bnb}_AGC_optimality_{independent|correlated}_{connected|disconnected}_merged.csv
```

Each merged AGC file has **25 rows** (5 m × 5 seeds).

### Usage

From `ODWB/`:

```bash
# E-optimal design (Boscia + SCIPSDP)
julia merge_single_runs_to_csv.jl
julia merge_single_runs_to_csv.jl Boscia
julia merge_single_runs_to_csv.jl SCIPSDP

# Boscia smoothing experiments (4 regimes)
julia merge_single_runs_to_csv.jl BosciaSmoothing

# AGC (Boscia + SCIPSDP, connected + disconnected)
julia merge_single_runs_to_csv.jl AGC
```

If no arguments are provided, the script runs all relevant groups (`Boscia`, `SCIPSDP`, `BosciaSmoothing`).

---

## 2. `aggregate_merged_by_group.jl`

### Purpose

Aggregate the merged CSVs into compact **summary CSVs** that combine solvers and/or regimes and compute:

- geometric mean of time,
- standard deviation of time w.r.t. geometric mean,
- number / percentage of solved instances,
- geometric mean of relative gap for unsolved instances,
- number of failed/missing instances,
- average LMO calls and nodes (Boscia / smoothing),
- average nodes, cuts, SDP iterations (SCIPSDP).

### Metrics

For each aggregated cell (solver + group), the script computes:

- `n_instances`: number of rows in the group,
- `n_solved`: count of `termination` ∈ {`OPTIMAL`, `GAPLIMIT`, `OPTIMALITY_PROVED`} with `time < TIME_LIMIT`,
- `pct_solved`: `100 * n_solved / n_instances`,
- `time_geom_mean`: geometric mean of positive finite `time`,
- `time_std_wrt_geom`: standard deviation of times around `time_geom_mean`,
- `rel_gap_geom_mean_unsolved`: geometric mean of positive finite `rel_gap` over unsolved rows,
- `failed_instances`: count of placeholder failures (from merge),
- `avg_lmo_calls`: mean `ncalls` over solved instances (Boscia and smoothing regimes only),
- `avg_nodes`: mean `nodes` over solved instances (all solvers),
- `avg_cuts`: mean `n_cuts_applied` over solved instances (SCIPSDP oa only),
- `avg_sdp_iters`: mean `n_sdp_iters` over solved instances (SCIPSDP bnb only).

All averages are rounded for readability.

### Modes and outputs

The script has three logical modes, controlled by flags:

1. **Default (no flags): EOPT Boscia + SCIPSDP**

   - Reads merged EOPT CSVs:

     ```text
     csv/Boscia/boscia_E_optimality_{independent|correlated}_cont_merged.csv
     csv/SCIPSDP/scip_sdp_{oa|bnb}_E_optimality_{independent|correlated}_cont_merged.csv
     ```

   - Combines all solvers into tables per **data type** (independent / correlated).
   - Aggregates in two ways per data type:

     - by **dimension** (`dimension = m`),
     - by **N_construction** (`rank_deficient`, `one`, `log`).

   - Outputs (**4 files**):

     ```text
     csv/aggregated/independent_by_dimension.csv
     csv/aggregated/independent_by_N_construction.csv
     csv/aggregated/correlated_by_dimension.csv
     csv/aggregated/correlated_by_N_construction.csv
     ```

2. **Smoothing mode: Boscia smoothing regimes**

   - Enabled with `--smoothing`.
   - Reads `boscia_smoothing_*_merged.csv` (4 regimes × 2 data types).
   - Treats each regime as a separate “solver” (`large_mu`, `small_mu`, `decay_0.9`, `decay_0.7`).
   - Same aggregation pattern as above (by dimension and by N_construction).

   - Outputs (**4 files**):

     ```text
     csv/aggregated/smoothing_independent_by_dimension.csv
     csv/aggregated/smoothing_independent_by_N_construction.csv
     csv/aggregated/smoothing_correlated_by_dimension.csv
     csv/aggregated/smoothing_correlated_by_N_construction.csv
     ```

3. **AGC mode: AGC Boscia + SCIPSDP**

   - Enabled with `--agc`.
   - Reads merged AGC CSVs:

     ```text
     csv/Boscia/boscia_AGC_optimality_{independent|correlated}_{connected|disconnected}_merged.csv
     csv/SCIPSDP/scip_sdp_{oa|bnb}_AGC_optimality_{independent|correlated}_{connected|disconnected}_merged.csv
     ```

   - For each connectivity (connected / disconnected) and data type (independent / correlated), combines solvers and aggregates **by dimension** only.
   - (For AGC, `N_construction` is always treated as `"other"`.)

   - Outputs (**4 files**, names chosen to make AGC explicit), e.g.:

     ```text
     csv/aggregated/agc_independent_connected_by_dimension.csv
     csv/aggregated/agc_independent_disconnected_by_dimension.csv
     csv/aggregated/agc_correlated_connected_by_dimension.csv
     csv/aggregated/agc_correlated_disconnected_by_dimension.csv
     ```

### Usage

From `ODWB/`:

```bash
# E-optimal design (Boscia + SCIPSDP)
julia aggregate_merged_by_group.jl

# Boscia smoothing regimes (4)
julia aggregate_merged_by_group.jl --smoothing

# AGC (Boscia + SCIPSDP, connected vs disconnected)
julia aggregate_merged_by_group.jl --agc
```

You can also pass `--out DIR` to change the output directory (defaults to `csv/aggregated`).

---

## 3. `aggregated_to_tex.jl`

### Purpose

Convert aggregated CSVs into **LaTeX `tabular` environments**, ready to include in the paper.

The script:

- produces **transposed** tables:
  - columns = dimension or N_construction,
  - rows = solver (or regime) × metric (via `\multirow`),
- highlights **best performance per column**:
  - bold highest `% sol.`,
  - bold lowest `time (s)`,
- allows **hiding** small columns (e.g. avg nodes),
- writes only the `tabular` (no surrounding `table` environment),
- writes `.tex` files directly into the paper repo.

Default paper output directory:

```text
/Users/deborah/Documents/research_projects/Smoothing-in-Boscia/paper
```

### Modes and inputs

The script distinguishes three sources of aggregated CSVs:

1. **EOPT mode (default, no `--smoothing` and no `--agc`)**

   - Reads from `csv/aggregated`:

     ```text
     independent_by_dimension.csv
     independent_by_N_construction.csv
     correlated_by_dimension.csv
     correlated_by_N_construction.csv
     ```

   - Solvers (row blocks):
     - `Boscia`
     - `SCIPSDP_oa`
     - `SCIPSDP_bnb`

2. **Smoothing mode (`--smoothing`)**

   - Reads from `csv/aggregated`:

     ```text
     smoothing_independent_by_dimension.csv
     smoothing_independent_by_N_construction.csv
     smoothing_correlated_by_dimension.csv
     smoothing_correlated_by_N_construction.csv
     ```

   - Regimes (treated as solvers in rows):
     - `large_mu`  → “Large $\mu$ (decay=1)”
     - `small_mu`  → “Small $\mu$ (decay=1)”
     - `decay_0.9` → “Decay 0.9”
     - `decay_0.7` → “Decay 0.7”

3. **AGC mode**

   - Not yet wired in explicitly, but easy to extend in the same style:
     - point to `agc_*_by_dimension.csv`,
     - set `solvers = ["Boscia", "SCIPSDP_oa", "SCIPSDP_bnb"]`,
     - call the same table writer.

### Metrics and highlighting

Columns (per row block) are drawn from the aggregated CSVs; the script knows about:

- `pct_solved` (`% sol.`, best = **max**),
- `time_geom_mean` (`time (s)`, best = **min**),
- `time_std_wrt_geom` (`time std`),
- `rel_gap_geom_mean_unsolved` (`rel. gap`),
- `failed_instances` (`failed`),
- `avg_lmo_calls` (`LMO`),
- `avg_nodes` (`nodes`),
- `avg_cuts` (`cuts`),
- `avg_sdp_iters` (`SDP it.`).

For each metric and **each column** (dimension or N_construction), it precomputes the best numeric value across solvers/regimes and then:

- prints `\textbf{...}` for entries equal (up to small tolerance) to the best value,
- e.g. bold lowest `time (s)` or highest `% sol.`.

### Hiding columns

You can hide any subset of metric columns with `--hide`. Examples:

```bash
# Hide all solver-specific stats, show only %sol, time, rel. gap, failed
julia aggregated_to_tex.jl --hide avg_lmo_calls avg_nodes avg_cuts avg_sdp_iters

# In smoothing mode
julia aggregated_to_tex.jl --smoothing --hide avg_lmo_calls avg_nodes avg_cuts avg_sdp_iters
```

The argument `--hide` accepts a list of names (space- or comma-separated).

### Output files

For EOPT:

```text
Smoothing-in-Boscia/paper/independent_by_dimension.tex
Smoothing-in-Boscia/paper/independent_by_N_construction.tex
Smoothing-in-Boscia/paper/correlated_by_dimension.tex
Smoothing-in-Boscia/paper/correlated_by_N_construction.tex
```

For smoothing:

```text
Smoothing-in-Boscia/paper/smoothing_independent_by_dimension.tex
Smoothing-in-Boscia/paper/smoothing_independent_by_N_construction.tex
Smoothing-in-Boscia/paper/smoothing_correlated_by_dimension.tex
Smoothing-in-Boscia/paper/smoothing_correlated_by_N_construction.tex
```

For AGC:

```text
Smoothing-in-Boscia/paper/agc_correlated_connected_by_dimension.tex
Smoothing-in-Boscia/paper/agc_independent_disconnected_by_dimension.tex
```

### Usage

From `ODWB/`:

```bash
# EOPT tables
julia aggregated_to_tex.jl --hide avg_lmo_calls avg_nodes avg_cuts avg_sdp_iters

# Smoothing tables (4 regimes)
julia aggregated_to_tex.jl --smoothing --hide avg_lmo_calls avg_nodes avg_cuts avg_sdp_iters

# AGC tables (2 setups: correlated_connected, independent_disconnected)
julia aggregated_to_tex.jl --agc --hide avg_lmo_calls avg_nodes avg_cuts avg_sdp_iters

# Custom output directory
julia aggregated_to_tex.jl --out /path/to/tex/out
```

At the end, the script prints the minimal LaTeX preamble:

```latex
\usepackage{booktabs}
\usepackage{siunitx}
\usepackage{multirow}
```

You can then include the tables in your main document with:

```latex
\begin{table}
  \centering
  \input{smoothing_independent_by_dimension}
  \caption{...}
  \label{tab:smoothing-independent-dimension}
\end{table}
```

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
