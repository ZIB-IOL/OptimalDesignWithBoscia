# E-Optimal Design CBF Export Script

This script exports E-optimal design models from the Pajarito formulation to CBF (Conic Benchmark Format) files for solving with external solvers.

## Overview

The script `export_eopt_to_cbf.jl` generates E-optimal design models for various dimensions and random seeds, then exports them to CBF format. The generated files can be solved with any CBF-compatible conic optimization solver.

## Features

- **Multiple Problem Sizes**: Generate models for different numbers of experiments (m) and parameters (n)
- **Multiple Random Seeds**: Create different problem instances with various random seeds
- **E-Optimal Criteria**: Support for both standard E-optimal ("E") and fusion E-optimal ("EF") designs
- **Data Types**: Support for both correlated and independent data generation
- **Organized Output**: Files are automatically organized with descriptive names

## Usage

### Basic Usage

```julia
julia --project=ODWB export_eopt_to_cbf.jl
```

This runs the script with default parameters, generating CBF files for:
- Dimensions: (20,5), (30,8), (50,10), (80,15)
- Seeds: 1, 2, 3, 4, 5
- Criteria: "E", "EF"
- Correlations: false, true

### Custom Usage

You can modify the script or use it programmatically:

```julia
include("export_eopt_to_cbf.jl")

# Define your parameters
dimensions = [(10, 3), (20, 5)]  # (m, n) pairs
seeds = [1, 2, 3]
criteria = ["E"]  # or ["E", "EF"]
correlations = [false]  # or [false, true]

# Generate CBF files
generate_cbf_files(
    dimensions, seeds, criteria, correlations,
    output_dir="my_cbf_models"
)
```

## Output Structure

The script creates CBF files with descriptive names:
```
eopt_{criterion}_{correlation}_{data_type}_m{m}_n{n}_seed{seed}.cbf
```

Where:
- `criterion`: "E" or "EF"
- `correlation`: "corr" or "indep"
- `data_type`: "cont" or "int" (for continuous or integer data)
- `m`: number of experiments
- `n`: number of parameters
- `seed`: random seed

Example: `eopt_E_indep_cont_m20_n5_seed1.cbf`

## Model Formulation

The E-optimal design problem is formulated as:

**Variables:**
- `x[1:m]`: Integer variables representing experiment counts
- `t`: Scalar variable (objective)

**Constraints:**
- `sum(x) >= N` and `sum(x) <= N`: Total experiment constraint
- `0 <= x <= ub`: Bounds on experiment counts
- `A' * diag(x) * A + t*I ⪰ 0`: E-optimality constraint (PSD)

**Objective:**
- Maximize `t` (maximize the minimum eigenvalue)

## Requirements

### Julia Packages
- JuMP
- Pajarito
- MathOptInterface
- Random, Printf, LinearAlgebra, Distributions, StableRNGs

### ODWB Module
The script requires the ODWB module from this project, which provides:
- `build_data()`: Data generation functions
- `build_integer_data()`: Integer data generation
- Other utility functions

## CBF Format Compatibility

The script uses MathOptInterface bridges to ensure compatibility with CBF format:
- Equality constraints are converted to pairs of inequalities
- PSD constraints are handled via bridges
- Integer variables are properly exported

## Solving the Generated Models

The CBF files can be solved with any CBF-compatible solver, such as:
- MOSEK
- SCS
- ECOS
- Clarabel
- And others supporting CBF format

Example with MOSEK:
```bash
mosek model.cbf
```

## File Sizes

Typical file sizes:
- Small problems (m=10, n=3): ~2-4 KB
- Medium problems (m=30, n=8): ~10-20 KB  
- Large problems (m=80, n=15): ~50-100 KB

## Notes

- The script automatically creates output directories
- Failed exports are logged with error messages
- The script handles both continuous and integer data generation
- All models use the same mathematical formulation as the Pajarito implementation

## Troubleshooting

If you encounter issues:

1. **Import Errors**: Ensure you're running with `--project=ODWB`
2. **CBF Export Failures**: Some constraint types may not be supported; check error messages
3. **Memory Issues**: For very large problems, consider reducing the problem size or running in batches

## Example Output

```
Generating 4 CBF models...
Output directory: cbf_eopt_models
============================================================
[1/4] Generating: eopt_E_indep_m20_n5_seed1.cbf
✓ Successfully exported model to: cbf_eopt_models/eopt_E_indep_m20_n5_seed1.cbf
  Parameters: m=20, n=5, N=7.0, criterion=E
  Variables: 21
  Upper bounds range: [1.0, 2.0]
...
============================================================
CBF file generation complete!
Files saved in: cbf_eopt_models
```

