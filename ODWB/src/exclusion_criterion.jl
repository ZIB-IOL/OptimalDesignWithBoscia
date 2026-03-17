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

function model_exclusion(A, m, n, UB, LB, M; u=fill(1.0, m), x=fill(0.0, m), vdix=-1)
    fixed_indices = Int64[]
    opt = optimizer_with_attributes(Mosek.Optimizer, MOI.Silent() => true)
    model = Model(opt)
    @variable(model, Z[1:n, 1:n])
    @constraint(model, sum(Z[i,i] for i in 1:n) == 1)
    @constraint(model, Z in PSDCone())
    for i in 1:m
        @constraint(model, M * A[i, :]' * Z * A[i, :] <= UB)
    end
    @show UB
   
    for i in 1:m
        if isapprox(x[i], 0.0, atol=1e-6) || u[i] == 0.0 || i == vdix
            continue
        end
        objective = M * A[i, :]' * Z * A[i, :] 
        set_objective_and_solve!(model, MOI.MAX_SENSE, objective)
        obj = objective_value(model)
        #@show i, obj, LB
        if obj <= LB
            @show i, obj, LB
            push!(fixed_indices, i)
        end
    end
   # @show fixed_indices
    return model, fixed_indices
end

function build_exclusion_branch_callback(A, N, f, sub_grad!)
    return function branch_callback(tree,
        node, vdix)
        m, n = size(A)
        if node.depth > n
            return false, false
        end
        u = fill(1.0, m)
        for i in tree.root.problem.integer_variables
            local_ub = get(node.local_bounds.upper_bounds, i, Inf)
            local_lb = get(node.local_bounds.lower_bounds, i, -Inf)
            if local_ub == 0.0 || local_lb == 1.0
                u[i] = 0.0
            end
        end
        y = copy(node.active_set.x/N) 
        @show vdix
        @show y
        Y = A' * diagm(y) * A
        λ, V = eigen(Y)
        if !isreal(λ[1])
            return false, false
        end
        λ_min = minimum(λ)
        tolerance = max(1e-10 * abs(λ_min), 1e-10)
        mult = count(λ_i -> abs(λ_i - λ_min) <= tolerance, λ)
        fx = -f(y)
        W = Symmetric(sum(V[:, j] * V[:, j]' for j in 1:mult)) #+ I(n)
        W -= n/2 * minimum(eigvals(W)) * I(n)
        W = 1/LinearAlgebra.tr(W) * W 

        @show mult

        UB = maximum(A[j,:]' * V[:,k] * V[:,k]' * A[j,:] for k in 1:mult for j in 1:m)

        if UB < fx
            return false, false
        end

        #UB = maximum(A[j,:]' * W * A[j,:] for j in 1:m) # * N
        _, fixed_indices = model_exclusion(A, m, n, UB, fx, 1.0, u=u, x=y, vdix=vdix)

        @show fixed_indices

        for i in fixed_indices
            push!(node.local_bounds.upper_bounds, (i => 0.0))
            #node.local_bounds.upper_bounds[i] = 0.0
        end
        return false, false
    end
end

function build_tightened_branch_callback(A, N, f, sub_grad!; L=nothing)
    return function branch_callback(tree,
        node, vdix)
        m, n = size(A)
        if node.depth > m
            return false, false
        end

        # collecting current fixings
        l = zeros(m)
        u = ones(m)
        fixed = []
        N_star = Int(N)
        for i in tree.root.problem.integer_variables
            local_ub = get(node.local_bounds.upper_bounds, i, Inf)
            local_lb = get(node.local_bounds.lower_bounds, i, -Inf)
            l[i] = isfinite(local_lb) ? local_lb : 0.0
            u[i] = isfinite(local_ub) ? local_ub : 1.0
            if isfinite(local_lb) || isfinite(local_ub)
                push!(fixed, i)
            end
            N_star = isfinite(local_lb) ? N_star - 1 : N_star
        end

        # generate dual z
        y = node.active_set.x
        Y = L === nothing ? A' * diagm(y) * A : L + A' * diagm(y) * A
        λ, V = eigen(Y)
        if !isreal(λ[1])
            return false, false
        end
        λ_min = minimum(λ)
        @show λ_min
        @show tree.root.options[:original_objective](y)
        tolerance = max(1e-10 * abs(λ_min), 1e-10)
        mult = count(λ_i -> abs(λ_i - λ_min) <= tolerance, λ)
        @show mult
        Z = Symmetric(sum(V[:, j] * V[:, j]' for j in 1:mult)) #+ I(n)
        Z = Z/tr(Z)
        v = [A[i,:]' * Z * A[i,:] for i in setdiff(1:m, fixed)]
        v = sort(v)
        t = v[Int(N_star)] 
      #= k = 0
        t = Inf
        for j in 1:mult 
            Z_temp = V[:, j] * V[:, j]'
            Z_temp = Z_temp/tr(Z_temp)
            v_temp = [A[i,:]' * Z_temp * A[i,:] for i in setdiff(collect(1:m), fixed)]
            v_temp = sort(v_temp)
            t_temp = v_temp[Int(N_star)]
            #@show t1
            #@show v1
            if t_temp < t
                t = t_temp
                k = j
            end
        end

       Z = V[:, k] * V[:, k]'
        Z = Z/tr(Z) =#
        # pick dual t (only for free variables)
        v = [A[i,:]' * Z * A[i,:] for i in 1:m]

        # compute α and β
        α = max.(0, t - v[i] for i in 1:m)
        β = max.(0, v[i] - t for i in 1:m)
        # perform the dual tightening
        UB = L === nothing ? t * N - α' * l + β' * u : t * N - α' * l + β' * u + dot(Z, L)
        @assert UB > λ_min " UB: $(UB) λ_min: $(λ_min)"
        if UB < -tree.incumbent
            return false, false
        end

        prune_left = false
        prune_right = false

        @show tree.incumbent
        @show UB
        fixed_to_one = []
        fixed_to_zero = []

        for i in tree.root.problem.integer_variables
            if i in fixed || i == vdix
                continue
            end
            right_bound = UB - α[i]
            left_bound = UB - β[i]
            @show left_bound, right_bound
            if right_bound <= -tree.incumbent 
                push!(node.local_bounds.upper_bounds, (i => 0.0))
                prune_right = i == vdix ? true : false
                push!(fixed_to_zero, i)
            elseif left_bound <= -tree.incumbent 
                push!(node.local_bounds.lower_bounds, (i => 1.0))
                prune_left = i == vdix ? true : false
                push!(fixed_to_one, i)
            end
        end
        @show fixed_to_one
        @show fixed_to_zero

        @assert length(fixed_to_one) < N
        @assert length(fixed_to_zero) < m - N
        return prune_left, prune_right
    end
end