"""
    set_objective!(model, sense, objective_expr; [solve=false])

Replace the objective of an existing JuMP model without rebuilding variables or constraints.
Useful when solving many problems that share the same constraint set and differ only in the objective.

- `model`: JuMP model (with variables and constraints already added).
- `sense`: `MOI.MIN_SENSE` or `MOI.MAX_SENSE`.
- `objective_expr`: JuMP expression (e.g. a variable ref, or `sum(c[i]*x[i] for i in 1:n)`).
- `solve`: if `true`, call `optimize!(model)` after setting the objective.

Returns the model. If `solve=true`, you can then use `termination_status(model)`, `objective_value(model)`, `value.(variables)`, etc.

# Example (SDP with fixed constraints, varying objective)

```julia
using JuMP
using Hypatia
const MOI = MathOptInterface

# Build model once: variables and constraints
opt = optimizer_with_attributes(Hypatia.Optimizer, MOI.Silent() => true)
model = Model(opt)
@variable(model, λ)
@variable(model, Z[1:n, 1:n], PSD)
@constraint(model, sum(Z[i,i] for i in 1:n) == 1)
# ... add all λ ≥ v_i' Z v_i constraints ...

# Solve with different objectives without rebuilding
for (name, obj_expr) in [("min λ", λ), ("min 2λ", 2λ)]
    set_objective!(model, MOI.MIN_SENSE, obj_expr; solve=true)
    @show name, objective_value(model)
end
```
    set_objective_and_solve!(model, sense, objective_expr)

Convenience: same as `set_objective!(model, sense, objective_expr; solve=true)`.
"""
function set_objective_and_solve!(
    model::JuMP.Model,
    sense::MOI.OptimizationSense,
    objective_expr,
)
    JuMP.set_objective_sense(model, sense)
    JuMP.set_objective_function(model, objective_expr)
    JuMP.optimize!(model)
    return model
end

function model_exclusion(A, m, n, UB, LB, M; u=fill(1.0, m), x=fill(0.0, m))
    fixed_indices = Int64[]
    opt = optimizer_with_attributes(Mosek.Optimizer, MOI.Silent() => true)
    model = Model(opt)
    @variable(model, Z[1:n, 1:n])
    @constraint(model, sum(Z[i,i] for i in 1:n) == 1)
    @constraint(model, Z in PSDCone())
    for i in 1:m
        @constraint(model, M * A[i, :]' * Z * A[i, :] <= UB)
    end
   
    for i in 1:m
        if x[i] != 0.0 || u[i] == 0.0
            continue
        end
        objective = M * A[i, :]' * Z * A[i, :] 
        set_objective_and_solve!(model, MOI.MAX_SENSE, objective)
        obj = objective_value(model)
        #@show i, obj, LB
        if obj <= LB
            push!(fixed_indices, i)
        end
    end
   # @show fixed_indices
    return model, fixed_indices
end
