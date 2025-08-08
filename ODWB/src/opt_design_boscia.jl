############## A optimal design ######################################################################
# Problem described here: https://link.springer.com/article/10.1007/s11222-014-9476-y
# "A first-order algorithm for the A-optimal experimental design Problem: a mathematical programming approach"

# min 1/(trace(∑x_i v_iv_i^T))
# s.t. \sum x_i = s
#       lb ≤ x ≤ ub
#       x ∈ Z^m

# v_i ∈ R^n
# n - number of parameters
# m - number of possible experiments
# A = [v_1^T,.., v_m^T], so the rows of A correspond to the different experiments


################################ D optimal design ########################################################################
# Problem described here: https://arxiv.org/pdf/2302.07386.pdf
# "Branch-and-Bound for D-Optimality with fast local search and bound tightening"

# min log(1/(det(∑x_i v_iv_i^T)))
# s.t. \sum x_i = s
#       lb ≤ x ≤ ub
#       x ∈ Z^m

# v_i ∈ R^n
# n - number of parameters
# m - number of possible experiments
# A = [v_1^T,.., v_m^T], so the rows of A correspond to the different experiments

################################ D-fusion design ########################################################################
# Problem described here: https://arxiv.org/pdf/2302.07386.pdf
# "Branch-and-Bound for D-Optimality with fast local search and bound tightening"

# min log(1/(det(∑x_i v_iv_i^T)))
# s.t. \sum x_i = s
#       lb ≤ x ≤ ub
#       x ∈ Z^m

# v_i ∈ R^n
# n - number of parameters
# m - number of possible experiments
# A = [v_1^T,.., v_m^T], so the rows of A correspond to the different experiments

function solve_opt(
    seed, 
    m, 
    n, 
    time_limit, 
    criterion, 
    corr; 
    full_callback=true, 
    p=0, 
    write = true, 
    verbose = true, 
    use_scip=false, 
    do_strong_branching=false, 
    use_shadow_set=false, 
    lazy_tolerance=2.0, 
    use_heuristics=false, 
    use_tightening=false, 
    long_runs=false, 
    options_run=false, 
    fw_verbose=false, 
    specific_seed=false,
    smoothing_start=1.0,
    smoothing_min=1e-3,
    smoothing_min_valid=false,
    smoothing_decay=0.9,
    integer_data=false,
    M=5,
)
    
    if criterion == "AF" || criterion == "DF"
        A, C, N, ub, _ = build_data(seed, m, n, true, corr; scaling_C=long_runs && criterion != "AF" && criterion != "DF")
    elseif criterion in ["E", "EF"]
        if integer_data
            A, C, N, ub, _ = build_integer_data(seed, m, n, criterion == "EF", corr; scaling_C=long_runs, M=M)
        else
            A, C, N, ub, _ = build_data(seed, m, n, criterion == "EF", corr; scaling_C=long_runs)
        end
    else
        A, _, N, ub, _ = build_data(seed, m, n, false, corr; scaling_C=long_runs)
    end

    # parameter tunning
    if !options_run
        use_heuristics = true
        if !(criterion in ["D","DF"])
            use_shadow_set = true
        elseif !(criterion in ["A","AF"])
            lazy_tolerance = 1.5
        end
    end

    if long_runs
        use_heuristics = true
        if criterion in ["A","AF"]
            use_shadow_set = true
        elseif criterion in ["D","DF"]
            lazy_tolerance = 1.5
        end
    end

    if use_scip
        o = SCIP.Optimizer()
        lmo, x = build_lmo(o, m, N, ub)
        branching_strategy = Bonobo.MOST_INFEASIBLE()
        heu = Boscia.Heuristic()
    else
        lmo = build_blmo(m, N, ub)
        if do_strong_branching
            function perform_strong_branch(tree, node)
                return node.level <= length(tree.root.problem.integer_variables) / 3
            end
            branching_strategy = Boscia.HybridStrongBranching(10, 1e-3, lmo, perform_strong_branch)
        else
            branching_strategy = Bonobo.MOST_INFEASIBLE()
        end

        if use_heuristics
            hyperplane_aware_rounding_prob = 0.8
            follow_gradient_prob=0.7
            follow_gradient_steps=n
            rounding_lmo_01_prob= criterion in ["E","EF"] ? 0.8 : 0.0
            probability_rounding_prob= criterion in ["E","EF"] ? 0.8 : 0.0
        else
            hyperplane_aware_rounding_prob = 0.0
            follow_gradient_prob=0.0
            follow_gradient_steps=n
            rounding_lmo_01_prob=0.0
            probability_rounding_prob=0.0
        end
    end
    result = 0.0
    domain_oracle = build_domain_oracle(A, n)

    println("build function")
    if criterion == "A"
        f, grad! = build_a_criterion(A, false, μ=1e-4, build_safe=true, long_run=long_runs)
        #f, grad! = build_general_trace(A, -1, false, build_safe=true)  # Log(Tr(X^{-1}))
    elseif criterion == "AF"
        f, grad! = build_a_criterion(A, true, C=C, long_run=long_runs)
        #f, grad! = build_general_trace(A, -1, true, C=C) # Log(Tr(X^{-1}))
    elseif criterion =="D" 
        f, grad! = build_d_criterion(A, false, μ=1e-4, build_safe=true, long_run=long_runs)
    elseif criterion == "DF"
        f, grad! = build_d_criterion(A, true, C=C, long_run=long_runs)
    elseif criterion == "E"
        f, generate_smoothing_function = build_e_criterion(A)
    elseif criterion == "EF"
        f, generate_smoothing_function = build_e_criterion(A)
    else
        error("Invalid criterion!")
    end


    if criterion in ["AF","DF"]
        direction = collect(1.0:m)
        x0 = compute_extreme_point(lmo, direction)
        active_set= FrankWolfe.ActiveSet([(1.0, x0)])   
        # set same incumbent as for Co-BnB
        z = greedy_incumbent_fusion(A,m,n,N,ub)

        # Precompile
        x, _, result = Boscia.solve(f, grad!, lmo; 
        settings_bnb = Boscia.settings_bnb(verbose=false, time_limit=10, active_set=active_set, branching_strategy=branching_strategy, use_shadow_set=use_shadow_set, start_solution=z),
        settings_tightening = Boscia.settings_tightening(dual_tightening=use_tightening, global_dual_tightening=use_tightening, lazy_tolerance=lazy_tolerance),
        settings_frank_wolfe = Boscia.settings_frank_wolfe(fw_verbose=fw_verbose, lazy_tolerance=lazy_tolerance),
        settings_heuristics = Boscia.settings_heuristics(hyperplane_aware_rounding_prob=hyperplane_aware_rounding_prob, follow_gradient_prob=follow_gradient_prob, follow_gradient_steps=follow_gradient_steps, rounding_lmo_01_prob=rounding_lmo_01_prob, probability_rounding_prob=probability_rounding_prob),
        )
        # Actual Run
        x, _, result = Boscia.solve(f, grad!, lmo; 
        settings_bnb = Boscia.settings_bnb(verbose=false, time_limit=time_limit, active_set=active_set, branching_strategy=branching_strategy, use_shadow_set=use_shadow_set, start_solution=z),
        settings_tightening = Boscia.settings_tightening(dual_tightening=use_tightening, global_dual_tightening=use_tightening, lazy_tolerance=lazy_tolerance),
        settings_frank_wolfe = Boscia.settings_frank_wolfe(fw_verbose=fw_verbose, lazy_tolerance=lazy_tolerance),
        settings_heuristics = Boscia.settings_heuristics(hyperplane_aware_rounding_prob=hyperplane_aware_rounding_prob, follow_gradient_prob=follow_gradient_prob, follow_gradient_steps=follow_gradient_steps, rounding_lmo_01_prob=rounding_lmo_01_prob, probability_rounding_prob=probability_rounding_prob),
        )
    elseif criterion in ["E", "EF"]
        line_search = FrankWolfe.Adaptive()
        # Precompile run
        x, _, result = Boscia.solve(f, nothing, lmo; 
            mode = Boscia.SMOOTHING_MODE,
            settings_bnb = Boscia.settings_bnb(verbose=true, time_limit=10, use_shadow_set=use_shadow_set, branching_strategy=branching_strategy),
            settings_smoothing = Boscia.settings_smoothing(mode=Boscia.SMOOTHING_MODE, generate_smoothing_objective = generate_smoothing_function, smoothing_start=smoothing_start, smoothing_min=smoothing_min, smoothing_min_valid=smoothing_min_valid, smoothing_decay=smoothing_decay),
            settings_frank_wolfe = Boscia.settings_frank_wolfe(mode=Boscia.SMOOTHING_MODE, max_fw_iter=1000, line_search=line_search, fw_verbose=fw_verbose, lazy_tolerance=lazy_tolerance),
            settings_tightening = Boscia.settings_tightening(dual_tightening=use_tightening, global_dual_tightening=use_tightening),
            settings_heuristics = Boscia.settings_heuristics(hyperplane_aware_rounding_prob=hyperplane_aware_rounding_prob, follow_gradient_prob=follow_gradient_prob, follow_gradient_steps=follow_gradient_steps, rounding_lmo_01_prob=rounding_lmo_01_prob, probability_rounding_prob=probability_rounding_prob),
        )
        # Actual run
        x, _, result = Boscia.solve(f, nothing, lmo; 
            mode = Boscia.SMOOTHING_MODE,
            settings_bnb = Boscia.settings_bnb(verbose=true, time_limit=time_limit, use_shadow_set=use_shadow_set, branching_strategy=branching_strategy),
            settings_smoothing = Boscia.settings_smoothing(mode=Boscia.SMOOTHING_MODE, generate_smoothing_objective = generate_smoothing_function, smoothing_start=smoothing_start, smoothing_min=smoothing_min, smoothing_min_valid=smoothing_min_valid, smoothing_decay=smoothing_decay),
            settings_frank_wolfe = Boscia.settings_frank_wolfe(mode=Boscia.SMOOTHING_MODE, max_fw_iter=1000, line_search=line_search, fw_verbose=fw_verbose, lazy_tolerance=lazy_tolerance),
            settings_tightening = Boscia.settings_tightening(dual_tightening=use_tightening, global_dual_tightening=use_tightening),
            settings_heuristics = Boscia.settings_heuristics(hyperplane_aware_rounding_prob=hyperplane_aware_rounding_prob, follow_gradient_prob=follow_gradient_prob, follow_gradient_steps=follow_gradient_steps, rounding_lmo_01_prob=rounding_lmo_01_prob, probability_rounding_prob=probability_rounding_prob),
        )
    else
        _, active_set, S = build_start_point2(A, m, n, N, ub)
        z = greedy_incumbent(A, m, n, N, ub)

        # Precompile
        x, _, result = Boscia.solve(f, grad!, lmo; 
        settings_bnb = Boscia.settings_bnb(verbose=false, time_limit=10, start_solution=z, branching_strategy=branching_strategy, use_shadow_set=use_shadow_set),
        settings_tightening = Boscia.settings_tightening(dual_tightening=use_tightening, global_dual_tightening=use_tightening),
        settings_frank_wolfe = Boscia.settings_frank_wolfe(fw_verbose=fw_verbose, lazy_tolerance=lazy_tolerance),
        settings_heuristics = Boscia.settings_heuristics(hyperplane_aware_rounding_prob=hyperplane_aware_rounding_prob, follow_gradient_prob=follow_gradient_prob, follow_gradient_steps=follow_gradient_steps, rounding_lmo_01_prob=rounding_lmo_01_prob, probability_rounding_prob=probability_rounding_prob),
        settings_domain = Boscia.settings_domain(domain_oracle=domain_oracle, active_set=active_set),
        )
        
        _, active_set, S = build_start_point2(A, m, n, N, ub)
        z = greedy_incumbent(A, m, n, N, ub)

        # Actual run
        x, _, result = Boscia.solve(f, grad!, lmo; 
        settings_bnb = Boscia.settings_bnb(verbose=verbose, time_limit=time_limit, start_solution=z, branching_strategy=branching_strategy, use_shadow_set=use_shadow_set),
        settings_tightening = Boscia.settings_tightening(dual_tightening=use_tightening, global_dual_tightening=use_tightening),
        settings_frank_wolfe = Boscia.settings_frank_wolfe(fw_verbose=fw_verbose, lazy_tolerance=lazy_tolerance),
        settings_heuristics = Boscia.settings_heuristics(hyperplane_aware_rounding_prob=hyperplane_aware_rounding_prob, follow_gradient_prob=follow_gradient_prob, follow_gradient_steps=follow_gradient_steps, rounding_lmo_01_prob=rounding_lmo_01_prob, probability_rounding_prob=probability_rounding_prob),
        settings_domain = Boscia.settings_domain(domain_oracle=domain_oracle, active_set=active_set),
        ) 
    end

    total_time_in_sec=result[:total_time_in_sec]
    status = result[:status]
    if occursin("Optimal", result[:status])
        status = "OPTIMAL"
    end
    if occursin("Time", result[:status])
        status = "TIME_LIMIT"
    end
    if full_callback
        lb_list = result[:list_lb]
        ub_list = result[:list_ub]
        time_list = result[:list_time]
        list_lmo_calls = result[:list_lmo_calls_acc]
        list_local_tightening = result[:local_tightenings]
        list_global_tightening = result[:global_tightenings]
    end

    if write
        folder = if long_runs
            "long_runs"
        elseif options_run
            if use_scip
                "MIP_SCIP"
            elseif do_strong_branching
                "strong_branching"
            elseif  use_shadow_set
                "shadow_set"
            elseif lazy_tolerance != 2.0
                "tighten_lazification"
            elseif use_heuristics
                "heuristics"
            elseif use_tightening
                "tightening"
            else
                "default"
            end
        else
            ""
        end

        df_full_cb = DataFrame(seed=seed, numberOfExperiments=m, numberOfParameters=n, N=N, time=time_list, lowerBound=lb_list, upperBound =ub_list, termination=status, LMOcalls =list_lmo_calls, localTighteings=list_local_tightening, globalTightenings=list_global_tightening)
        file_name_full_cb = joinpath(@__DIR__, "../csv/full_runs_boscia/" * folder * "/boscia_"* criterion * "_optimality_" * type *"_" * string(m) * "-" * string(n) * "_" * string(seed) * ".csv")
        CSV.write(file_name_full_cb, df_full_cb, append=false)

        # CSV file for the results of all instances.
        scaled_solution = result[:primal_objective]*m
        df = DataFrame(seed=seed, numberOfExperiments=m, numberOfParameters=n, N=N, time=total_time_in_sec, solution=result[:primal_objective], scaled_solution=scaled_solution, dual_gap = result[:dual_gap],  rel_dual_gap=result[:rel_dual_gap], ncalls=result[:lmo_calls], num_nodes=result[:number_nodes],termination=status)
        file_name = joinpath(@__DIR__, "../csv/Boscia/" * folder * "/boscia_" * criterion * "_" * string(m) * "_" * type *"_optimality.csv")
        if !isfile(file_name) 
            CSV.write(file_name, df, append=true, writeheader=true, delim=";")
        else 
            CSV.write(file_name, df, append=true, delim=";")
        end
    end

    if x !== nothing 
        # check feasibility
        if use_scip
            o = SCIP.Optimizer()
            check_lmo,_ = build_lmo(o, m, N, ub)
            @show Boscia.is_linear_feasible(check_lmo, x) 
        else
            check_lmo = build_blmo(m, N, ub)
            @show Boscia.is_linear_feasible(check_lmo, x)
        end
        @show result[:primal_objective]
        @show result[:dual_gap]
        @show result[:primal_objective] - result[:dual_gap]
        @show x
        @show sum(x)
        @show N
    end

    return x, result
end
