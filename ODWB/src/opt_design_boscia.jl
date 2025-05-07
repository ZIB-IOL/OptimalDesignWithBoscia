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
    write=true, 
    verbose=true, 
    use_scip=false, 
    do_strong_branching=false, 
    use_shadow_set=false, 
    lazy_tolerance=2.0, 
    use_heuristics=false, 
    use_tightening=false, 
    long_runs=false, 
    options_run=false, 
    fw_verbose=false, 
    ls_secant=false, 
    sharpness=false, 
    log_trace=false, 
    zero_one=false,
    use_BCG=false,
    print_iter=1
)

    type = corr ? "correlated" : "independent"
    
    if criterion in ["AF","DF","GTIF"]
        A, C, N, ub, _ = build_data(seed, m, n, true, corr; scaling_C=long_runs && criterion != "AF" && criterion != "DF")
    else
        A, _, N, ub, _ = build_data(seed, m, n, false, corr; scaling_C=long_runs, zero_one=zero_one)
    end

    # parameter tunning
    if !options_run
        use_heuristics = true
        ls_secant = true
        use_tightening = true
        use_shadow_set = true
        #use_BCG=true
    end

    if long_runs
        use_heuristics = true
        ls_secant = true
        use_tightening = true
        use_shadow_set = true
    end

    ls_secant = log_trace ? true : ls_secant

    if use_scip
        o = SCIP.Optimizer()
        lmo, x = build_lmo(o, m, N, ub, silent=true)
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
            heu = Boscia.Heuristic(Boscia.rounding_hyperplane_heuristic, 0.8, :hyperplane_rounding)
        else
            heu = Boscia.Heuristic()
        end
    end

    result = 0.0
    domain_oracle = build_domain_oracle(A, n)
    domain_point = build_domain_point_function(domain_oracle, A, N, collect(1:m), fill(0.0, m), ub)

    println("build function")
    if criterion == "A"
        f, grad! = build_a_criterion(A, false, μ=long_runs ? 1e-3 : 1e-4, build_safe=false, long_run=long_runs)
        #f, grad! = build_general_trace(A, -1, false, build_safe=!ls_secant)
    elseif criterion == "AF"
        f, grad! = build_a_criterion(A, true, C=C, long_run=long_runs)
        #f, grad! = build_general_trace(A, -1, true, C=C)
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


    if criterion in ["AF","DF","GTIF"]
        direction = collect(1.0:m)
        println("find first extreme point")
        x0 = compute_extreme_point(lmo, direction)
        line_search = ls_secant ? FrankWolfe.Secant() : FrankWolfe.Adaptive()
        fw_variant = use_BCG ? Boscia.Blended() : Boscia.BPCG()
        active_set= FrankWolfe.ActiveSet([(1.0, x0)])   
        @show f(x0) 
        z = greedy_incumbent_fusion(A,m,n,N,ub)

        x, _, result = Boscia.solve(f, grad!, lmo; verbose=false, time_limit=10, active_set=active_set, branching_strategy=branching_strategy, use_shadow_set=use_shadow_set, dual_tightening=use_tightening, global_dual_tightening=use_tightening, lazy_tolerance=lazy_tolerance, custom_heuristics=[heu], start_solution=z, line_search=line_search, variant=fw_variant)

        x, _, result = Boscia.solve(f, grad!, lmo; verbose=verbose, time_limit=time_limit, active_set=active_set, branching_strategy=branching_strategy, use_shadow_set=use_shadow_set, dual_tightening=use_tightening, global_dual_tightening=use_tightening, lazy_tolerance=lazy_tolerance, custom_heuristics=[heu], fw_verbose=fw_verbose, start_solution=z,line_search=line_search, variant=fw_variant, print_iter=print_iter)
    else
        # Precompile
        _, active_set, S = build_start_point2(A, m, n, N, ub)
        line_search = ls_secant ? FrankWolfe.Secant(domain_oracle=domain_oracle) : FrankWolfe.MonotonicGenericStepsize(FrankWolfe.Adaptive(), domain_oracle)
        fw_variant = use_BCG ? Boscia.Blended() : Boscia.BPCG()
        z = greedy_incumbent(A, m, n, N, ub)
        x, _, result = Boscia.solve(f, grad!, lmo; verbose=false, time_limit=10, variant=fw_variant, active_set=active_set, domain_oracle=domain_oracle, find_domain_point=domain_point, start_solution=z, dual_tightening=use_tightening, global_dual_tightening=use_tightening, lazy_tolerance=lazy_tolerance, branching_strategy=branching_strategy, use_shadow_set=use_shadow_set, custom_heuristics=[heu], line_search=line_search, print_iter=1, fw_verbose=false) #line_search=StepSizeRule, 
        
        # real run
        # Find a good start point
        _, active_set, S = build_start_point2(A, m, n, N, ub)
        # initial upper bound
        z = greedy_incumbent(A, m, n, N, ub)
        line_search = ls_secant ? FrankWolfe.Secant(domain_oracle=domain_oracle) : FrankWolfe.MonotonicGenericStepsize(FrankWolfe.Adaptive(), domain_oracle)
        fw_variant = use_BCG ? Boscia.Blended() : Boscia.BPCG()
        x, _, result = Boscia.solve(f, grad!, lmo; verbose=verbose, time_limit=time_limit, variant=fw_variant, active_set=active_set, domain_oracle=domain_oracle, find_domain_point=domain_point, start_solution=z,  dual_tightening=use_tightening, global_dual_tightening=use_tightening, lazy_tolerance=lazy_tolerance, branching_strategy=branching_strategy, use_shadow_set=use_shadow_set, custom_heuristics=[heu], fw_verbose=fw_verbose, line_search=line_search, print_iter=print_iter) #line_search=StepSizeRule, 
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

    if log_trace
        f_check, _ = criterion in ["AF", "GTIF"] ? build_general_trace(A, p, true, C=C) : build_general_trace(A, p, false)
    else
        f_check = f
    end
    scaled_solution = x !== nothing ? f_check(x) : Inf
    @show scaled_solution

    if write
        subfolder = if long_runs
            "long_runs"
        elseif log_trace
            "log_trace"
        elseif criterion in ["GTI", "GTIF"]
            "GTI"
        elseif !options_run
            ""
        elseif use_heuristics
            "heuristics"
        elseif use_scip
            "mip_scip"
        elseif sharpness
            "sharpness"
        elseif use_shadow_set
            "shadow_set"
        elseif do_strong_branching
            "strong_branching"
        elseif use_tightening
            "tightening"
        elseif lazy_tolerance != 2.0
            "tighten_lazification"
        elseif use_BCG
            "blended"
        elseif ls_secant
            "secant"
        else
            "default"
        end

        if criterion in ["GTI","GTIF"]
            criterion = criterion * "_" * string(Int64(p*100))
        end
           
        if full_callback
            lb_list = result[:list_lb]
            ub_list = result[:list_ub]
            time_list = result[:list_time]
            list_lmo_calls = result[:list_lmo_calls_acc]
            list_active_set_size_cb = result[:list_active_set_size] 
            list_discarded_set_size_cb = result[:list_discarded_set_size]
            list_local_tightening = result[:local_tightenings]
            list_global_tightening = result[:global_tightenings]
            df = DataFrame(seed=seed, dimension=n, time=time_list, lowerBound= lb_list, upperBound = ub_list, termination=status, LMOcalls = list_lmo_calls, localTighteings=list_local_tightening, globalTightenings=list_global_tightening, list_active_set_size_cb=list_active_set_size_cb,list_discarded_set_size_cb=list_discarded_set_size_cb)
            file_name = joinpath(@__DIR__, "../csv/full_runs_boscia/" * subfolder * "/boscia_" * criterion * "_optimality_" * type * "_" * string(m) * "_" * string(n) * "_" * string(seed) * ".csv" )
            CSV.write(file_name, df, append=false)
        end

        df = DataFrame(seed=seed, numberOfExperiments=m, numberOfParameters=n, N=N, time=total_time_in_sec, solution=result[:primal_objective], scaled_solution=scaled_solution, dual_gap = result[:dual_gap],  rel_dual_gap=result[:rel_dual_gap], ncalls=result[:lmo_calls], num_nodes=result[:number_nodes],termination=status)
        
        file_name = joinpath(@__DIR__, "../csv/Boscia/" * subfolder * "/boscia_" * criterion * "_" * type * "_" * string(m) * "_" * string(n) * "_" * string(seed) * ".csv")

        CSV.write(file_name, df, append=false, writeheader=true)
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
