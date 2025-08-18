# Minimal Working Example: E-Optimal Design with Pajarito
# This file solves the E-optimal design problem using Pajarito solver

using JuMP
using Pajarito
using HiGHS
using Hypatia
using LinearAlgebra
using Random
using Distributions
using StableRNGs

# Data generation function (adapted from utilities.jl)
function build_data(seed, m, n, fusion, corr; scaling_C=false, zero_one=false)
    # Set up random seed
    Random.seed!(seed)
    
    if corr 
        B = rand(m, n)
        B = B' * B
        @assert isposdef(B)
        D = MvNormal(randn(n), B)
        A = rand(D, m)'
        @assert rank(A) == n 
    else 
        A = rand(m, n)
        @assert rank(A) == n # check that A has the desired rank!
    end 
    
    C_hat = rand(2n, n)
    C = scaling_C ? 1/(2n) * transpose(C_hat) * C_hat : transpose(C_hat) * C_hat
    @assert rank(C) == n
    
    if fusion
        N = rand(floor(m/20):floor(m/3))
        ub = rand(1.0:m/10, m)
    else
        N = floor(1.5 * n)
        u = floor(N/3)
        ub = rand(1.0:u, m)
    end
        
    if zero_one
        return A, C, N, fill(1.0, m), C_hat
    end

    return A, C, N, ub, C_hat
end

# E-criterion function (adapted from utilities.jl)
function build_e_criterion(A)
    m, n = size(A)
    
    function inf_matrix(x)
        return Symmetric(A' * diagm(x) * A)
    end

    function f(x)
        X = inf_matrix(x)   
        return (-1) * minimum(eigvals(X))    
    end

    return f
end

# Build E-optimal Pajarito model (based on build_E_pajarito_model)
function build_E_pajarito_model_minimal(seed, m, n, criterion, time_limit, corr, verbose=true)
    # Generate data
    if criterion == "EF"
        A, C, N, ub, _ = build_data(seed, m, n, true, corr)
    else
        A, _, N, ub, _ = build_data(seed, m, n, false, corr)
    end

    println("Problem size: m=$m, n=$n, N=$N, sum(ub)=$(sum(ub))")

    # Setup solvers
    # MIP solver 
    oa_solver = optimizer_with_attributes(HiGHS.Optimizer,
        MOI.Silent() => !verbose,
        "mip_feasibility_tolerance" => 1e-8,
        "mip_rel_gap" => 1e-6,
    )
    
    # SDP solver
    conic_solver = optimizer_with_attributes(Hypatia.Optimizer, 
        MOI.Silent() => !verbose,
    )
    
    # Pajarito optimizer
    opt = optimizer_with_attributes(Pajarito.Optimizer,
        "time_limit" => time_limit, 
        "iteration_limit" => 100000,
        "oa_solver" => oa_solver, 
        "conic_solver" => conic_solver,
        "tol_rel_gap" => 5e-2,
        "tol_abs_gap" => 1e-6,
        MOI.Silent() => !verbose,
    )

    # Create model
    model = Model(opt)
    
    # Add variables
    @variable(model, x[1:m])
    set_integer.(x)
    @variable(model, t)
    
    # Constraints
    @constraint(model, sum(x) == N)
    @constraint(model, x in MOI.Nonnegatives(m))
    @constraint(model, x .<= ub)
    
    # Objective: maximize t (minimize negative largest eigenvalue)
    @objective(model, Max, t)

    # PSD constraint: A' * diag(x) * A + t*I ⪰ 0
    # This ensures that the smallest eigenvalue of A' * diag(x) * A is >= -t
    if criterion == "E"
        # Information matrix: A' * diag(x) * A - t*I ⪰ 0
        info_matrix = [
            @expression(model, 
                (i == j ? -t : 0.0) + sum(A[k, i] * x[k] * A[k, j] for k in 1:m)
            ) for i in 1:n, j in 1:n
        ]
        # Add PSD constraint
        @constraint(model, info_matrix in PSDCone())
    elseif criterion == "EF"
        # For fusion case, use C matrix as well
        info_matrix = [
            @expression(model, 
                C[i, j] + (i == j ? -t : 0.0) + sum(A[k, i] * x[k] * A[k, j] for k in 1:m)
            ) for i in 1:n, j in 1:n
        ]
        # Add PSD constraint  
        @constraint(model, info_matrix in PSDCone())
    end

    return model, x, t, A, (criterion == "EF" ? C : zeros(n, n)), N, ub
end

# Main solving function
function solve_e_optimal_pajarito_minimal(seed=1, m=50, n=7, time_limit=300, criterion="E", corr=false, verbose=true)
    println("=== E-Optimal Design with Pajarito ===")
    println("Parameters: seed=$seed, m=$m, n=$n, criterion=$criterion, corr=$corr")
    
    # Build model
    model, x, t, A, C, N, ub = build_E_pajarito_model_minimal(seed, m, n, criterion, time_limit, corr, verbose)
    
    # Solve the model
    println("\nSolving...")
    optimize!(model)
    
    # Extract results
    status = termination_status(model)
    solution_x = value.(x)
    solution_t = value(t)
    solve_time_sec = solve_time(model)
    
    # Get additional solver info
    paja_opt = JuMP.unsafe_backend(model)
    num_iterations = paja_opt.num_iters
    num_cuts = paja_opt.num_cuts
    
    println("\n=== Results ===")
    println("Status: $status")
    println("Optimal t value: $solution_t")
    println("Solve time: $(solve_time_sec) seconds")
    println("Number of iterations: $num_iterations")
    println("Number of cuts: $num_cuts")
    println("Solution x (first 10 components): $(solution_x[1:min(10, length(solution_x))])")
    println("Sum of x: $(sum(solution_x))")
    
    # Check feasibility
    println("\n=== Feasibility Check ===")
    println("Sum constraint satisfied: $(abs(sum(solution_x) - N) < 1e-6)")
    println("Non-negativity satisfied: $(all(solution_x .>= -1e-6))")
    println("Upper bound satisfied: $(all(solution_x .<= ub .+ 1e-6))")
    
    # Compute actual E-criterion value
    f = build_e_criterion(A)
    actual_e_value = f(solution_x)
    println("Actual E-criterion value: $actual_e_value")
    println("Theoretical lower bound (-t): $(-solution_t)")
    
    return solution_x, solution_t, status, solve_time_sec
end

# Example usage
if abspath(PROGRAM_FILE) == @__FILE__
    # Run a small example
    println("Running minimal E-optimal design example...")
    solution_x, solution_t, status, solve_time_sec = solve_e_optimal_pajarito_minimal(
        1,      # seed
        30,     # m (number of experiments)
        Int(floor(sqrt(30))),      # n (number of parameters)  
        300,    # time_limit
        "E",    # criterion
        false,  # corr (uncorrelated data)
        true    # verbose
    )
    
    println("\nExample completed successfully!")
end
