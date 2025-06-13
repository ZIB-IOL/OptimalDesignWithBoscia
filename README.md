# Optimal Experiment Design with Boscia

The repository for the "Solving the Optimal Experiment Design Problem with Mixed-Integer Convex Methods" paper.

The experiment scripts are in the ODWB folder. The models and set-ups for the different solvers can also be found in the source folder in the ODWB folder. 

The models and solvers tested are [Boscia.jl](https://github.com/ZIB-IOL/Boscia.jl) with the classical problem formulation, [SCIP.jl](https://github.com/scipopt/SCIP.jl) using the epigraph formulation, [Pajarito.jl](https://github.com/jump-dev/Pajarito.jl) implementing the direct conic model and a custom solver based from this [paper](https://link.springer.com/article/10.1007/s11222-021-10043-5)

The experiments are set up to be run on a cluster. If you want to test instances of one dimension with a specific solver, you can use Julia's environmental variables and call the ```run_optimal_design.jl``` file directly.

```
using ODWB

# This sets the solver. Option are "Boscia", "SCIP", "Pajarito" and "Custom".
ENV["MODE"] = "Boscia"
# Set the criterion. Options are "A", "D" for the A-Optimal Design and D-Optimal Design, respectively.
# "AF" and "DF" are for the A-Fusion and D-Fusion. 
ENV["CRITERION"] = "D"
# This decides the data type, so independent ("IND") or correlated data ("CORR"). 
ENV["TYPE"] = "CORR"
# Sets the dimension of the problem.
ENV["DIMENSION"] = "100"

include("run_optimal_design.jl")
```

For individual instances, you can call the solvers directly.

```
using ODWB

seed = 1
m = 100
n = 10
criterion = "D"
time_limit = 3600 # 1 hour
corr = true # this create correlated data, if false, it will create independent data.

# Boscia
ODWB.solve_opt(seed, m, n, time_limit, criterion,corr)

# SCIP
if criterion in ["A", "D"]
    error("SCIP OA does not work with the optimal problems!")
end
ODWB.solve_opt_scip(seed, m, n, time_limit, criterion, corr)

# Pajarito
ODWB.solve_opt_pajarito(seed, m, n, time_limit, criterion, corr)

# Custom
ODWB.solve_opt_custom(seed, m, n, time_limit, criterion, corr)
```
