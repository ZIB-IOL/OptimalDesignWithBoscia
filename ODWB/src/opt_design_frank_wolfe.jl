# Optimal design with FrankWolfe
function solve_opt_frank_wolfe(
    seed, 
    m, 
    n, 
    time_limit, 
    criterion, 
    corr; 
    write=true, 
    verbose=true, 
    use_scip=false, 
    options_run=false,
    ls_secant=true,#
    long_runs=false,
    log_trace=false,
    p=1,
    zero_one=false
)
    type = corr ? "correlated" : "independent"
    
    if criterion in ["AF","DF","GTIF"]
        A, C, N, ub, _ = build_data(seed, m, n, true, corr; scaling_C=long_runs && criterion != "AF" && criterion != "DF")
    else
        A, _, N, ub, _ = build_data(seed, m, n, false, corr; scaling_C=long_runs, zero_one=zero_one)
    end

    solution = 0.0
    lmo = build_blmo(m, N, ub)
    result = 0.0
    domain_oracle = build_domain_oracle(A, n)

    println("build function")
    if criterion == "A"
        f, grad! = if log_trace 
            build_a_criterion(A, false, μ=long_runs ? 1e-3 : 1e-4, build_safe=false, long_run=long_runs)
        else
            build_general_trace(A, 1, false, build_safe=!ls_secant)
        end
    elseif criterion == "AF"
        f, grad! = build_a_criterion(A, true, C=C, long_run=long_runs)
    elseif criterion =="D" 
        f, grad! = build_d_criterion(A, false, μ=1e-4, build_safe=false, long_run=long_runs)
    elseif criterion == "DF"
        f, grad! = build_d_criterion(A, true, C=C, long_run=long_runs)
    elseif criterion == "GTI"
        f, grad! = log_trace ? build_general_log_trace(A, p, false) : build_general_trace(A, p, false)
    elseif criterion == "GTIF"
        f, grad! = log_trace ? build_general_log_trace(A, p, true, C=C) : build_general_trace(A, p, true, C=C)
    else
        error("Invalid criterion!")
    end

    if criterion == "AF" || criterion == "DF" || occursin("GTIF", criterion)
        direction = collect(1.0:m)
        x0 = compute_extreme_point(lmo, direction)

        x, _, primal, dual_gap, _, _ = FrankWolfe.blended_pairwise_conditional_gradient(f, grad!, lmo, x0; timeout=10, lazy=true)
        (x, _, primal, dual_gap, traj_data, _), time,_,_,_ = @timed FrankWolfe.blended_pairwise_conditional_gradient(f, grad!, lmo, x0; timeout=time_limit, lazy=true, trajectory = true, max_iteration = 1000000)

    else
        line_search = FrankWolfe.Secant(domain_oracle=domain_oracle)
        _, active_set, S = build_start_point2(A, m, n, N, ub)
        x, _, primal, dual_gap, _, _ = FrankWolfe.blended_pairwise_conditional_gradient(f, grad!, lmo, active_set; timeout=10, lazy=true, line_search=line_search)

        _, active_set, S = build_start_point2(A, m, n, N, ub)
        (x, _, primal, dual_gap, traj_data, _), time,_,_,_ = @timed FrankWolfe.blended_pairwise_conditional_gradient(f, grad!, lmo, active_set; timeout=time_limit, lazy=true, trajectory = true, max_iteration = 1000000, line_search=line_search, verbose = verbose)

    end

        iteration = traj_data[end][1]
        if dual_gap ≤ 1e-7
            status = "OPTIMAL"
        elseif time ≥ limit
            status = "Time limit reached"
        elseif iteration ≥ 1000000
            status = "Iteration max out"
        end

        # Create integer solution and check feasibility
        k = length(findall(x->x!=0, x))
        o = SCIP.Optimizer()
            check_lmo,_ = build_lmo(o,m,N,ub)
        @show Boscia.is_linear_feasible(check_lmo.o, x)
        @show sum(x)
        @show primal
        @show x
        if k >= N
            solution_int = Inf
            feasible = false
        else
            x_int = heuristics(x, N, ub, mode)
            if x_int != Inf
                feasible = Boscia.is_linear_feasible(check_lmo.o, x_int)
                @assert sum(x_int) == N
                solution_int = f(x_int)
            else
                solution_int = Inf
                feasible = false
            end
        end

    type = corr ? "correlated" : "independent"
    criterion= log_trace && criterion == "A" ? "GTI_100" : criterion
        
    if write
        # trajectory
        df_traj = DataFrame(traj_data)
        rename!(df_traj, Dict(1 => "iterations", 2 => "primal", 3 => "dual_bound", 4 => "dual_gap", 5 => "time"))
        file_name_traj = joinpath(@__DIR__, "../csv/FrankWolfe/trajectory/" * criterion * "-" * type * "_" * string(m) * "_" * string(n) * "_" * string(seed) * ".csv")
        CSV.write(file_name_traj, df_traj, append=false, writeheader=true)
        

        # CSV file for the results of all instances.
        df = DataFrame(seed=seed, numberOfExperiments=m, numberOfParameters=n, N=N,time=time, solution_fw=primal, solution_int=solution_int, dual_gap = dual_gap, termination=status, nIteration=iteration, feasibilty=feasible)
        file_name = joinpath(@__DIR__, "../csv/FrankWolfe/" * criterion * "-" * type * "_" * string(m) * "_" * string(n) * "_" * string(seed) * ".csv")
        if !isfile(file_name) 
            CSV.write(file_name, df, append=true, writeheader=true, delim=";")
        else 
            CSV.write(file_name, df, append=true, delim=";")
        end
    end

    return x, solution
end

"""
Build the LMO for FW.
In the limit case:
    min g(A^T diag(x) A)
    s.t. sum x = 1
    x ∈ [0,1]

In the cont case:
    min g(A^T diag(x) A)
    s.t. sum x = s
    x ∈ [0,ub]
"""
function build_fw_lmo(o, m, s, ub)
    MOI.set(o, MOI.Silent(), true)
    MOI.empty!(o)
    x = MOI.add_variables(o, m)
    for i in 1:m
        MOI.add_constraint(o, x[i], MOI.GreaterThan(0.0))
        if mode == "limit"
            MOI.add_constraint(o, x[i], MOI.LessThan(1.0))
        elseif mode == "cont"
            MOI.add_constraint(o, x[i], MOI.LessThan(ub[i]))
        end
    end

    MOI.add_constraint(
        o,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(m), x), 0.0),
        MOI.LessThan(s)
    )
    MOI.add_constraint(
        o,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(m), x), 0.0),
        MOI.GreaterThan(1.0)
    )

    lmo = FrankWolfe.MathOptLMO(o)

    return lmo, x
end