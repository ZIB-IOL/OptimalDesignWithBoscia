# Solving opt design problem with Pajarito

# Pajarito model for the D-optimal problems
function build_D_pajarito_model(seed, m, n, criterion, time_limit, corr, verbose=true, zero_one=false)
    if criterion == "DF" 
        A, C, N, ub, _ = build_data(seed, m, n, true, corr, zero_one=zero_one)
    else
        A, _, N, ub, _ = build_data(seed, m, n, false, corr, zero_one=zero_one)
        @assert N ≥ n
    end
    @show m, n, N, sum(ub) 
    @assert (m > n) && (sum(ub) >= N)

    # setup solvers
    # MIP solver (try SCIP as well?)
    oa_solver = optimizer_with_attributes(HiGHS.Optimizer,
        MOI.Silent() => !verbose,
       # "mip_feasibility_tolerance" => 1e-8,
       # "mip_rel_gap" => 1e-6,
    )
    # SDP solver
    conic_solver = optimizer_with_attributes(Hypatia.Optimizer, 
        MOI.Silent() => !verbose,
    )
    opt = optimizer_with_attributes(Pajarito.Optimizer,
        "time_limit" => time_limit, 
        "iteration_limit" => 100000,
        "oa_solver" => oa_solver, 
        "conic_solver" => conic_solver,
        MOI.Silent() => !verbose,
    )

    model = Model(opt)
    # add variables
    JuMP.@variable(model, x[1:m], Int)
    # we want to do s experiments
    JuMP.@constraint(model, sum(x) == N)

    # Constraints on the total times each experiment can be run
    ub_u = copy(ub)
    unique!(ub_u)
    for u in ub_u
        ind = findall(x->x==u, ub)
        mid = u / 2
        JuMP.@constraint(model, vcat(mid, x[ind] .- mid) in MOI.NormInfinityCone(length(ind) + 1))
    end

    JuMP.@variable(model, t)

    # information matrix lower triangle
    if criterion == "D"
        a1 = [
            JuMP.@expression(model, sum(A[k, i] * x[k] * A[k, j] for k in 1:m)) for
            i in 1:n for j in 1:i
        ]
        JuMP.@constraint(model, vcat(t, 1, a1) in MOI.LogDetConeTriangle(n))
    elseif criterion == "DF"
        a1 = [
            JuMP.@expression(model, C[i,j] + sum(A[k, i] * x[k] * A[k, j] for k in 1:m)) for
            i in 1:n for j in 1:i
        ]
        JuMP.@constraint(model, vcat(t, 1, a1) in MOI.LogDetConeTriangle(n))
        
    end
    JuMP.@objective(model, Max, t)

    return model, x, t
end

# Pajarito model for the A-optimal problems
# As suggested here: https://github.com/jump-dev/Pajarito.jl/issues/444
function build_A_pajarito_model(seed, m , n, criterion, time_limit, corr, verbose = true, zero_one=false)
    #error("Pajarito and A-opt: Needs to be fixed!!")
    if criterion == "AF"
        A, C, N, ub = build_data(seed, m, n, true, corr, zero_one=zero_one)
    else
        A, _, N, ub = build_data(seed, m, n, false, corr, zero_one=zero_one)
        @assert N ≥ n
    end
    @assert (m > n) && (sum(ub) >= N)

    # setup solvers
    # MIP solver (try SCIP as well?)
    oa_solver = optimizer_with_attributes(HiGHS.Optimizer,
        MOI.Silent() => !verbose,
       # "mip_feasibility_tolerance" => 1e-8,
       # "mip_rel_gap" => 1e-6,
    )
    # SDP solver
    conic_solver = optimizer_with_attributes(Hypatia.Optimizer, 
        MOI.Silent() => !verbose,
    )
    opt = optimizer_with_attributes(Pajarito.Optimizer,
        "time_limit" => time_limit, 
        "iteration_limit" => 100000,
        "oa_solver" => oa_solver, 
        "conic_solver" => conic_solver,
        MOI.Silent() => !verbose,
    )

    model = Model(opt)
    # add variables
    JuMP.@variable(model, x[1:m])
    JuMP.set_integer.(x)
    JuMP.@variable(model, t)
    # we want to do s experiments
    JuMP.@constraint(model, sum(x) == N)
    @objective(model, Min, 4 * t)

    # Constraints on the total times each experiment can be run
    ub_u = copy(ub)
    unique!(ub_u)
    for u in ub_u
        ind = findall(x->x==u, ub)
        mid = u / 2
        JuMP.@constraint(model, vcat(mid, x[ind] .- mid) in MOI.NormInfinityCone(length(ind) + 1))
    end

    if criterion == "A"
        # vectorized information matrix
        X_vec = [
            JuMP.@expression(
                model,
                (i == j ? 1.0 : sqrt(2)) * sum(A[k, i] * x[k] * A[k, j] for k in 1:m)
            ) for i in 1:n for j in 1:i
        ]
        add_homog_spectral(MatNegSqrtConj() , n, vcat(1.0 * t, X_vec), model)
    elseif criterion == "AF"
        X_vec = [
            JuMP.@expression(
                model,
                (i == j ? 1.0 : sqrt(2)) * (C[i,j] + sum(A[k, i] * x[k] * A[k, j] for k in 1:m))
            ) for i in 1:n for j in 1:i
        ]
        add_homog_spectral(MatNegSqrtConj() , n, vcat(1.0 * t, X_vec), model)
    end

#=elseif criterion == "A"
    # https://discourse.julialang.org/t/how-to-optimize-trace-of-matrix-inverse-with-jump-or-convex/94167/4
    cone = EpiPerSepSpectralCone{Float64}(Hypatia.Cones.NegSqrtSSF(), Hypatia.Cones.MatrixCSqr{Float64, Float64}, n, true)
    @constraint(model, vcat(1.0, t, [ sum(A[k, i] * x[k] * A[k,j] for k in 1:m) * (i == j ? 1 : sqrt(2)) for i in 1:n for j in 1:i]...) in cone)
    @objective(model, Min, 4 * t)
elseif criterion == "AF"
    cone = EpiPerSepSpectralCone{Float64}(Hypatia.Cones.NegSqrtSSF(), Hypatia.Cones.MatrixCSqr{Float64, Float64}, n, true)
    @constraint(model, vcat(1.0, t, [ (C[i,j] + sum(A[k, i] * x[k] * A[k,j] for k in 1:m)) * (i == j ? 1 : sqrt(2)) for i in 1:n for j in 1:i]...) in cone)
    @objective(model, Min, 4 * t)
end=#

    return model, x, t
end

function build_E_pajarito_model(seed, m, n, criterion, time_limit, corr, verbose=true, integer_data=false, zero_one=false)
    if criterion == "EF"
        A, C, N, ub, _ = integer_data ? build_integer_data(seed, m, n, true, corr) : build_data(seed, m, n, true, corr, zero_one=zero_one)
    else
        A, _, N, ub, _ = integer_data ? build_integer_data(seed, m, n, false, corr) : build_data(seed, m, n, false, corr, zero_one=zero_one)
    end

    # setup solvers
    # MIP solver (try SCIP as well?)
    oa_solver = optimizer_with_attributes(HiGHS.Optimizer,
        MOI.Silent() => true, #!verbose,
        "mip_feasibility_tolerance" => 1e-8,
        "mip_rel_gap" => 1e-6,
    )
    # SDP solver
    conic_solver = optimizer_with_attributes(Hypatia.Optimizer, 
        MOI.Silent() => true, #!verbose,
    )
    opt = optimizer_with_attributes(Pajarito.Optimizer,
        "time_limit" => time_limit, 
        "iteration_limit" => 100000,
        "oa_solver" => oa_solver, 
        "conic_solver" => conic_solver,
        "tol_rel_gap" => 5e-2,
        "tol_abs_gap" => 1e-6,
        MOI.Silent() => !verbose,
    )

    model = Model(opt)
    # add variables
    JuMP.@variable(model, x[1:m])
    JuMP.set_integer.(x)
    JuMP.@variable(model, t)
    # we want to do s experiments
    JuMP.@constraint(model, sum(x) == N)
    @objective(model, Max, t)

    # Constraints on the total times each experiment can be run
    #=ub_u = copy(ub)
    unique!(ub_u)
    for u in ub_u
        ind = findall(x->x==u, ub)
        mid = u / 2
        JuMP.@constraint(model, vcat(mid, x[ind] .- mid) in MOI.NormInfinityCone(length(ind) + 1))
    end =#
    JuMP.@constraint(model, x in MOI.Nonnegatives(m))
    JuMP.@constraint(model, x <= ub)

    # PSD constraint: A' * diag(x) * A + t*I ⪰ 0
    # This is equivalent to: A' * diag(x) * A - (-t)*I ⪰ 0
    # We want to maximize t, so we minimize -t (the largest eigenvalue)
    if criterion == "E"
        # Information matrix: A' * diag(x) * A + t*I
        info_matrix = [
            JuMP.@expression(model, 
                (i == j ? -t : 0.0) + sum(A[k, i] * x[k] * A[k, j] for k in 1:m)
            ) for i in 1:n, j in 1:n
        ]
        # Add PSD constraint
        JuMP.@constraint(model, info_matrix in JuMP.PSDCone())
    elseif criterion == "EF"
        # For fusion case, use C matrix as well
        info_matrix = [
            JuMP.@expression(model, 
                C[i, j] + (i == j ? -t : 0.0) + sum(A[k, i] * x[k] * A[k, j] for k in 1:m)
            ) for i in 1:n, j in 1:n
        ]
        # Add PSD constraint  
        JuMP.@constraint(model, info_matrix in JuMP.PSDCone())
    end

    return model, x, t
end

function is_point_feasible(model, solution_dict)
    # Check all constraints by manually evaluating them
    feasible = true
    violations = []
    
    for constraint_ref in all_constraints(model, include_variable_in_set_constraints=true)
        constraint_obj = constraint_object(constraint_ref)
        func = constraint_obj.func
        set = constraint_obj.set
        
        # Manually evaluate the constraint function
        func_value = evaluate_constraint_function(func, solution_dict)
        
        # Check if the constraint is satisfied
        if !is_constraint_satisfied(func_value, set)
            feasible = false
            push!(violations, (constraint_ref, func_value, set))
            println("Constraint violated: ", constraint_ref)
            println("  Function value: ", func_value)
            println("  Set: ", set)
        end
    end
    
    return feasible, violations
end

# Helper function to manually evaluate constraint functions
function evaluate_constraint_function(func, solution_dict)
    if func isa JuMP.VariableRef
        # Single variable
        return solution_dict[func]
    elseif func isa JuMP.AffExpr
        # Affine expression: constant + sum(coeff * var)
        result = func.constant
        for (var, coeff) in func.terms
            result += coeff * solution_dict[var]
        end
        return result
    elseif func isa JuMP.QuadExpr
        # Quadratic expression: aff_part + sum(coeff * var1 * var2)
        result = evaluate_constraint_function(func.aff, solution_dict)
        for (var_pair, coeff) in func.terms
            var1, var2 = var_pair.a, var_pair.b
            result += coeff * solution_dict[var1] * solution_dict[var2]
        end
        return result
    elseif func isa Vector
        # Vector of expressions (for vector constraints)
        return [evaluate_constraint_function(f, solution_dict) for f in func]
    else
        error("Unsupported constraint function type: $(typeof(func))")
    end
end

# Helper function to check if a value satisfies a constraint set
function is_constraint_satisfied(value, set)
    # Handle different MOI set types
    if set isa MOI.EqualTo
        return isapprox(value, set.value, atol=1e-6)
    elseif set isa MOI.LessThan
        return value <= set.upper + 1e-6  # Small tolerance for numerical errors
    elseif set isa MOI.GreaterThan
        return value >= set.lower - 1e-6
    elseif set isa MOI.Interval
        return (value >= set.lower - 1e-6) && (value <= set.upper + 1e-6)
    elseif set isa MOI.ZeroOne
        return isapprox(value, 0.0, atol=1e-6) || isapprox(value, 1.0, atol=1e-6)
    elseif set isa MOI.Integer
        return isapprox(value, round(value), atol=1e-6)
    elseif set isa MOI.Zeros
        # For vector constraints - all elements should be zero
        return all(isapprox(v, 0.0, atol=1e-6) for v in value)
    elseif set isa MOI.Nonnegatives
        # All elements should be non-negative
        return all(v >= -1e-6 for v in value)
    elseif set isa MOI.Nonpositives
        # All elements should be non-positive
        return all(v <= 1e-6 for v in value)
    elseif set isa MOI.SecondOrderCone
        # ||x[2:end]|| <= x[1]
        if length(value) < 2
            return value[1] >= -1e-6
        else
            norm_tail = sqrt(sum(value[i]^2 for i in 2:length(value)))
            return norm_tail <= value[1] + 1e-6
        end
    elseif set isa MOI.PositiveSemidefiniteConeTriangle
        # Check if the matrix represented by the triangular vector is PSD
        return check_psd_constraint(value)
    else
        @warn "Unknown constraint set type: $(typeof(set)). Assuming satisfied."
        return true
    end
end

# Helper function to check PSD constraint for E-optimal design
function check_psd_constraint(matrix_values)
    # Convert the triangular vector back to a symmetric matrix
    # The triangular vector represents the upper triangle of the matrix in column-major order
    n = Int((-1 + sqrt(1 + 8*length(matrix_values))) / 2)  # Solve for matrix dimension
    
    # Reconstruct the symmetric matrix from triangular representation
    matrix = zeros(n, n)
    idx = 1
    for j in 1:n
        for i in 1:j
            matrix[i, j] = matrix_values[idx]
            if i != j
                matrix[j, i] = matrix_values[idx]  # Symmetric
            end
            idx += 1
        end
    end
    
    # Check if the matrix is positive semidefinite by computing eigenvalues
    try
        eigenvals_matrix = eigvals(Symmetric(matrix))
        min_eigenval = minimum(eigenvals_matrix)
        
        # Matrix is PSD if all eigenvalues are non-negative (with small tolerance)
        is_psd = min_eigenval >= -1e-8
        
        if !is_psd
            println("  PSD violation: minimum eigenvalue = $min_eigenval")
        end
        
        return is_psd
    catch e
        @warn "Error computing eigenvalues for PSD check: $e"
        return false
    end
end

# Specialized function to check the E-optimal constraint directly
function check_e_optimal_constraint(x_values, t_value, A)
    # The constraint is: A' * diag(x) * A + t*I ⪰ 0
    # This is equivalent to: minimum eigenvalue of (A' * diag(x) * A) ≥ -t
    
    m, n = size(A)
    
    # Compute information matrix: A' * diag(x) * A
    info_matrix = A' * diagm(x_values) * A
    
    # Compute eigenvalues
    try
        eigenvals_info = eigvals(Symmetric(info_matrix))
        min_eigenval = minimum(eigenvals_info)
        
        # Check if min_eigenval ≥ -t (with tolerance)
        constraint_satisfied = min_eigenval >= -t_value - 1e-8
        
        if !constraint_satisfied
            println("  E-optimal constraint violation:")
            println("    Minimum eigenvalue of A'*diag(x)*A: $min_eigenval")
            println("    -t value: $(-t_value)")
            println("    Required: min_eigenval ≥ -t")
        else
            println("  E-optimal constraint satisfied:")
            println("    Minimum eigenvalue: $min_eigenval")
            println("    -t value: $(-t_value)")
        end
        
        return constraint_satisfied
    catch e
        @warn "Error computing eigenvalues for E-optimal check: $e"
        return false
    end
end


function solve_opt_pajarito(seed, m, n, time_limit, criterion, corr; write=true, verbose=true, integer_data=false, boscia_solution=nothing, zero_one=false)
    if criterion == "DF" || criterion == "D"
        model, x, epi = build_D_pajarito_model(seed, m, n, criterion, 10, corr, false, zero_one)
        optimize!(model)
        model, x, epi = build_D_pajarito_model(seed, m, n, criterion, time_limit, corr, verbose, zero_one)
    elseif criterion == "AF" || criterion == "A"
        model, x, epi = build_A_pajarito_model(seed, m, n, criterion, 10, corr, false)
        optimize!(model)
        model, x, epi = build_A_pajarito_model(seed, m, n, criterion, time_limit, corr, verbose)
    elseif criterion == "E" || criterion == "EF"
        model, x, epi = build_E_pajarito_model(seed, m, n, criterion, 10, corr, false, integer_data)
        optimize!(model)
        model, x, epi = build_E_pajarito_model(seed, m, n, criterion, time_limit, corr, verbose, integer_data)
    end

    # solve 
    optimize!(model)

    # query solution
    status = termination_status(model)
    solution = objective_value(model)
    solution = criterion == "D" || criterion == "DF" ? solution * (-1) : solution
    y = value.(x)
    t = solve_time(model)
    paja_opt = JuMP.unsafe_backend(model)
    numberIter = paja_opt.num_cuts
    numberCuts = paja_opt.num_iters

    # Check feasibility
    if criterion == "A" || criterion == "D"
        A, C, N, ub, _ = build_data(seed, m, n, false, corr, zero_one=zero_one)
    elseif criterion == "AF"|| criterion == "DF"
        A, C, N, ub, _ = build_data(seed, m, n, true, corr, zero_one=zero_one)
    elseif criterion == "E" || criterion == "EF"
        A, C, N, ub, _ = integer_data ? build_integer_data(seed, m, n, true, corr) : build_data(seed, m, n, true, corr, zero_one=zero_one)
    else
        A, _, N, ub, _ = integer_data ? build_integer_data(seed, m, n, false, corr) : build_data(seed, m, n, false, corr, zero_one=zero_one)
    end
    if criterion in ["A","AF"]
        f_check, _ = build_a_criterion(A, criterion == "AF", C=C, build_safe = false, μ=criterion == "A" ? 1e-4 : 0.0)
    elseif criterion in ["GTI","GTIF"]
        f_check, _ = build_general_trace(A, p, criterion == "GTIF", C=C)
    elseif criterion == "E" || criterion == "EF"
        f_check, _ = build_e_criterion(A)
    else
        f_check, _ = build_d_criterion(A, criterion == "DF", C=C, build_safe = false, μ=criterion == "D" ? 1e-4 : 0.0)
    end
    feasible = isfeasible(seed, m, n,criterion, y, corr, ub=ub)
    @show feasible

    if boscia_solution !== nothing
        solution_dict = Dict(x[i] => boscia_solution[i] for i in eachindex(boscia_solution))
        merge!(solution_dict, Dict(epi => (-1) * f_check(boscia_solution)))
        
        println("=== Feasibility Check for Boscia Solution ===")
        feasible, violations = is_point_feasible(model, solution_dict)
        @show feasible
        @show violations
        
        # Additional specialized check for E-optimal constraint
        if criterion == "E" || criterion == "EF"
            println("\n=== Specialized E-optimal Constraint Check ===")
            t_value = f_check(boscia_solution)
            e_optimal_satisfied = check_e_optimal_constraint(boscia_solution, t_value, A)
            println("E-optimal constraint satisfied: $e_optimal_satisfied")
        end
    end

    # o = JuMP.moi_backend(model)
    type = corr ? "correlated" : "independent"
    scaled_solution = if feasible 
        f_check(y)
    else
        Inf 
    end
    @show status
    @show y
    @show solution
    @show scaled_solution

    if write 
        df = DataFrame(seed=seed, numberOfExperiments=m, numberOfParameters=n, time=t, N=N, solution=solution, scaled_solution=scaled_solution, termination=status, numberIterations=numberIter, numberCuts=numberCuts, feasible=feasible)
        file_name = joinpath(@__DIR__, "../csv/Pajarito/pajarito_" * criterion * "_optimality_" * type * "_" * string(m) * "_" * string(n) * "_" * string(seed) * ".csv" )
        CSV.write(file_name, df, append=false)
        #if !isfile(file_name)
        #    CSV.write(file_name, df, append=true, writeheader=true)
        #else 
        #    CSV.write(file_name, df, append=true)
        #end
    end
    #@assert feasible
    return y
end