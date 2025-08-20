# Minimal Working Example: E-Optimal Design with Pajarito
# This file solves the E-optimal design problem using Pajarito solver

using JuMP
using Pajarito
using HiGHS
using Hypatia
using LinearAlgebra

A = [0.640560125688334 0.8988924651705463 0.6026315293750658 0.09942414283811873; 0.3244420648938249 0.21960977858670605 0.8752427026103645 0.7330721267274701; 0.4284151183931746 0.06037385523635341 0.4479437123418444 0.40800090600302463; 0.3539981006442484 0.11431932418843094 0.6797418029178991 0.3676010970291791; 0.6240687276682546 0.8276760414998834 0.8951460038632557 0.9408622568595247; 0.6093405087159935 0.5475397233252607 0.3753804665727667 0.6283323077663837; 0.8148945670877987 0.8159334877118128 0.42139733688904235 0.4596363701822491; 0.8581761761029205 0.6791489434158131 0.8162784435702374 0.38054865157982964; 0.4037354612337182 0.9234073469913137 0.4925105187447588 0.4202196080406787; 0.0465632242904217 0.588608026827289 0.21073940556559967 0.9357195644138552; 0.46079375823300184 0.9655435124426841 0.33129739228282684 0.8749097390632607; 0.23653413851585414 0.9689717890441853 0.4285687872702951 0.9887116589634328; 0.5616823570640757 0.3700511410277585 0.7705341197413721 0.8850442312936958; 0.8295347567225702 0.9252731672804814 0.11200245309187484 0.654354549533481; 0.6877230841636806 0.36562522511027085 0.5380654173827643 0.33035378144723415; 0.48457722652079827 0.15673193660803542 0.20485906499945394 0.6362942989719393; 0.8574183603562568 0.5761171392212233 0.6599616466199261 0.5833842527846491; 0.5019688218144165 0.4815792427281611 0.5051499066121654 0.7951315576134046; 0.46509258174600643 0.8499021885218158 0.6089184177322454 0.1776254689978578; 0.9139243738764997 0.1512127891717162 0.8357702479228285 0.25057409400489294]
N = 6

# Build E-optimal Pajarito model (based on build_E_pajarito_model)
function build_E_pajarito_model_minimal(m, n, time_limit; verbose=true)
    # Generate data
    #A, N, ub = build_data(seed, m, n, zero_one=zero_one)
    ub = fill(1.0, m)
    
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

    # PSD constraint: A' * diag(x) * A - t*I ⪰ 0
    # This ensures that the smallest eigenvalue of A' * diag(x) * A is >= t
    # Information matrix: A' * diag(x) * A - t*I ⪰ 0
    info_matrix = [
        @expression(model, 
            (i == j ? -t : 0.0) + sum(A[k, i] * x[k] * A[k, j] for k in 1:m)
        ) for i in 1:n, j in 1:n
    ]
    # Add PSD constraint
    @constraint(model, info_matrix in PSDCone())


    return model, x, t, A, N, ub
end

# Main solving function
function solve_e_optimal_pajarito_minimal(;m=50, n=7, time_limit=300, verbose=true)
    println("=== E-Optimal Design with Pajarito ===")
    println("Parameters: m=$m, n=$n")
    
    # Build model
    model, x, t, A, N, ub = build_E_pajarito_model_minimal(m, n, time_limit, verbose=verbose)
    
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
    println("Solution x: $(solution_x[1:min(20, length(solution_x))])")
    println("Sum of x: $(sum(solution_x))")
    
    return solution_x, solution_t, status, solve_time_sec
end

# Example usage
if abspath(PROGRAM_FILE) == @__FILE__
    # Run a small example
    println("Running minimal E-optimal design example...")
    solution_x, solution_t, status, solve_time_sec = solve_e_optimal_pajarito_minimal(
        m=size(A, 1),     # m (number of experiments)
        n=size(A, 2),      # n (number of parameters)  
        time_limit=300,    # time_limit
        verbose=true,   # verbose
    )
    
    println("\n=== Compare with a different solver ===")
    println("Solution found with a different solver: 0.373251984739451")
    println("x = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0]")
end
