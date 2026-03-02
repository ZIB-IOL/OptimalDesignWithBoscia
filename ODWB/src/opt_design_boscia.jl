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
    ls_secant=false, 
    sharpness=false, 
    log_trace=false, 
    zero_one=true,
    use_BPCG=false,
    print_iter=1,
    specific_seed=false,
    smoothing_start=m/20,
    smoothing_min=max(exp10(-m/100), 1e-3),
    smoothing_min_valid=false,
    smoothing_decay=0.7,
    use_follow_subgradient_heu=false,
    use_pipage_heu=false,
    use_sr_rounding_heu=false,
    use_fedorov_heu=false,
    N=-Inf,
    M=5,
    use_exclusion_criterion=false,
    use_sub_grad_info=false,
    branch_all=false,
    connected=true,
    mu_testing=false
)
    type = corr ? "correlated" : "independent"
    
    if criterion in ["AF","DF","GTIF"]
        A, C, N, ub, _ = build_data(seed, m, n, true, corr; scaling_C=long_runs && criterion != "AF" && criterion != "DF")
    elseif criterion in ["E", "EF"]
        A, C, N, ub, _ = build_data(seed, m, n, criterion == "EF", corr; scaling_C=long_runs, N=N, zero_one=zero_one)
        L = nothing
    elseif criterion == "AGC"
        present_edges = connected ? Int(floor(2 * m)) : Int(floor(1/2 * m))
        edges, potential_edges = build_graph_connectivity_data(n, present_edges, m, seed=seed, connected=connected)
        L = graph_laplacian(n, edges) + ones(n, n)
        A = potential_edges_incidence_matrix(n, potential_edges)
        ub = fill(1.0, m)
        N = !isfinite(N) ? Int(floor(m/2)) : N
    else
        A, _, N, ub, _ = build_data(seed, m, n, false, corr; scaling_C=long_runs, zero_one=zero_one)
    end

    # parameter tunning
    if !options_run
        use_heuristics = true
        if criterion in ["E","EF","AGC"]
            use_tightening = false
            use_shadow_set = false
            use_sub_grad_info = true
            ls_secant = false
        elseif !(criterion in ["D","DF"])
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

    ls_secant = log_trace ? true : ls_secant

    if use_scip
        o = SCIP.Optimizer()
        lmo, x = build_lmo(o, m, N, ub, silent=true)
        branching_strategy = Bonobo.MOST_INFEASIBLE()
        heu = Boscia.Heuristic()

        hyperplane_aware_rounding_prob = 0.0
        follow_gradient_prob=0.3
        follow_gradient_steps=n
        rounding_lmo_01_prob=0.7
        probability_rounding_prob=0.7
        rounding_prob =0.3
        custom_heu = []
    else
        lmo = build_blmo(m, N, ub)
        custom_heu = []
        
        if use_follow_subgradient_heu || use_pipage_heu || use_sr_rounding_heu || use_fedorov_heu
            hyperplane_aware_rounding_prob = 0.0
            follow_gradient_prob=0.0
            follow_gradient_steps=n
            rounding_lmo_01_prob=0.0
            probability_rounding_prob=0.0
            rounding_prob =0.0
        else
            hyperplane_aware_rounding_prob = 0.8
            follow_gradient_prob=0.5
            follow_gradient_steps=n
            rounding_lmo_01_prob= criterion in ["E","EF","AGC"] ? 0.8 : 0.0
            probability_rounding_prob= criterion in ["E","EF","AGC"] ? 0.8 : 0.0
            rounding_prob =1.0
        end
        if use_heuristics
            if criterion in ["E","EF","AGC"]
                follow_subgradient_heuristic = build_follow_subgradient_heuristic(A, n, L=L)
                push!(custom_heu, Boscia.Heuristic(follow_subgradient_heuristic, 0.5, :follow_subgradient))
                sr_rounding_heuristic = build_simple_randomized_rounding_heuristic(A, N, 20)
                push!(custom_heu, Boscia.Heuristic(sr_rounding_heuristic, 1.0, :sr_rounding))
            end
            if N > 1.5 * n
                pipage_rounding_heuristic = build_pipage_rounding_heuristic(A, N, L=L)
                push!(custom_heu, Boscia.Heuristic(pipage_rounding_heuristic, criterion == "AGC" ? 0.0 : 0.3, :pipage_rounding))
            end
            fedorov_heuristic = build_greedy_fedorov_heuristic(A, N, 10, L=L)
            push!(custom_heu, Boscia.Heuristic(fedorov_heuristic, 0.4, :fedorov))
        elseif use_follow_subgradient_heu
            if criterion in ["E","EF","AGC"]
                follow_subgradient_heuristic = build_follow_subgradient_heuristic(A, n, L=L)
                push!(custom_heu, Boscia.Heuristic(follow_subgradient_heuristic, 0.5, :follow_subgradient))
            end
        elseif use_pipage_heu
            if N > 1.5 * n
                pipage_rounding_heuristic = build_pipage_rounding_heuristic(A, N, L=L)
                push!(custom_heu, Boscia.Heuristic(pipage_rounding_heuristic, 0.3, :pipage_rounding))
            end
        elseif use_sr_rounding_heu
            if criterion in ["E","EF","AGC"]
                sr_rounding_heuristic = build_simple_randomized_rounding_heuristic(A, N, 10)
                push!(custom_heu, Boscia.Heuristic(sr_rounding_heuristic, 1.0, :sr_rounding))
            end
        elseif use_fedorov_heu
            fedorov_heuristic = build_greedy_fedorov_heuristic(A, N, 10, L=L)
            push!(custom_heu, Boscia.Heuristic(fedorov_heuristic, 1.0, :fedorov))
        end
    end

    if do_strong_branching
        function perform_strong_branch(tree, node)
            return node.level <= length(tree.root.problem.integer_variables) / 3
        end
        branching_strategy = Boscia.HybridStrongBranching(10, 1e-3, lmo, perform_strong_branch)
    elseif branch_all
        branching_strategy = Boscia.BRANCH_ALL()
    else
        branching_strategy = Bonobo.MOST_INFEASIBLE()
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
    elseif criterion in ["E","AGC"]
        f, sub_grad!, generate_smoothing_function = build_e_criterion(A, L=L)
    elseif criterion == "EF"
        f, sub_grad!, generate_smoothing_function = build_e_criterion(A, L=C)
    elseif criterion == "GTI"
        f, grad! = log_trace ? build_general_log_trace(A, p, false) : build_general_trace(A, p, false)
    elseif criterion == "GTIF"
        f, grad! = log_trace ? build_general_log_trace(A, p, true, C=C) : build_general_trace(A, p, true, C=C)
    else
        error("Invalid criterion!")
    end

    line_search = if ls_secant
        if criterion in ["A", "D", "GTI"]
            FrankWolfe.Secant(domain_oracle=domain_oracle)
        else
            FrankWolfe.Secant()
        end
    else
        if criterion in ["A", "D", "GTI"]
            FrankWolfe.MonotonicGenericStepsize(FrankWolfe.Adaptive(), domain_oracle)
        else
            FrankWolfe.Adaptive()
        end
    end

    if use_exclusion_criterion && criterion in ["E", "EF", "AGC"]
        function build_bnb_callback(A, N, f, sub_grad!)
            return function bnb_callback(tree,
                node;
                worse_than_incumbent=false,
                node_infeasible=false,
                lb_update=false,)
                m, n = size(A)
                if node.depth > m/10
                    return
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
                Y = A' * diagm(y) * A
                λ, V = eigen(Y)
                if !isreal(λ[1])
                    return
                end
                λ_min = minimum(λ)
                tolerance = max(1e-10 * abs(λ_min), 1e-10)
                mult = count(λ_i -> abs(λ_i - λ_min) <= tolerance, λ)
                fx = -f(y)
                W = Symmetric(sum(V[:, j] * V[:, j]' for j in 1:mult)) #+ I(n)
                W -= n/2 * minimum(eigvals(W)) * I(n)
                W = 1/LinearAlgebra.tr(W) * W 
        
                UB = N * maximum(A[j,:]' * W * A[j,:] for j in 1:m)
                _, fixed_indices = model_exclusion(A, m, n, UB, fx, 1.0, u=u, x=y)

                @show fixed_indices
        
                for i in fixed_indices
                    node.local_bounds.upper_bounds[i] = 0.0
                end
            end
        end
        tree_callback = build_bnb_callback(A, N, f, sub_grad!)
    else
        tree_callback = nothing
    end

    fw_variant = use_BPCG ? Boscia.BlendedPairwiseConditionalGradient() : Boscia.DecompositionInvariantConditionalGradient()

    if criterion in ["AF","DF","GTIF"]
        direction = collect(1.0:m)
        println("find first extreme point")
        x0 = compute_extreme_point(lmo, direction)
        line_search = ls_secant ? FrankWolfe.Secant() : FrankWolfe.Adaptive()
        fw_variant = use_BCG ? Boscia.Blended() : Boscia.BPCG()
        active_set= FrankWolfe.ActiveSet([(1.0, x0)])   
        @show f(x0) 
        z = greedy_incumbent_fusion(A,m,n,N,ub)

        # Precompile
        settings = Boscia.create_default_settings()
        settings.branch_and_bound[:verbose] = false
        settings.branch_and_bound[:time_limit] = 10
        settings.branch_and_bound[:branching_strategy] = branching_strategy
        settings.branch_and_bound[:use_shadow_set] = use_shadow_set
        settings.branch_and_bound[:start_solution] = z
        settings.domain[:active_set] = active_set
        settings.tightening[:dual_tightening] = use_tightening
        settings.tightening[:global_dual_tightening] = use_tightening
        settings.tightening[:lazy_tolerance] = lazy_tolerance
        settings.frank_wolfe[:fw_verbose] = fw_verbose
        settings.frank_wolfe[:lazy_tolerance] = lazy_tolerance
        settings.frank_wolfe[:variant] = fw_variant
        settings.frank_wolfe[:line_search] = line_search
        settings.heuristic[:hyperplane_aware_rounding_prob] = hyperplane_aware_rounding_prob
        settings.heuristic[:follow_gradient_prob] = follow_gradient_prob
        settings.heuristic[:follow_gradient_steps] = follow_gradient_steps
        settings.heuristic[:rounding_lmo_01_prob] = rounding_lmo_01_prob
        settings.heuristic[:probability_rounding_prob] = probability_rounding_prob
        settings.heuristic[:rounding_prob] = rounding_prob
        settings.heuristic[:custom_heuristics] = custom_heu
        if tree_callback !== nothing
            settings.branch_and_bound[:bnb_callback] = tree_callback
        end
        x, _, result = Boscia.solve(f, grad!, lmo, settings=settings)
        
        # Actual Run
        settings.branch_and_bound[:time_limit] = time_limit
        x, _, result = Boscia.solve(f, grad!, lmo, settings=settings)
    elseif criterion in ["E", "EF", "AGC"]
        line_search = ls_secant ? FrankWolfe.Secant() : FrankWolfe.Adaptive()
        # Precompile run
        settings = Boscia.create_default_settings(mode=Boscia.SMOOTHING_MODE)
        settings.branch_and_bound[:verbose] = false
        settings.branch_and_bound[:time_limit] = 10
        settings.branch_and_bound[:use_shadow_set] = use_shadow_set
        settings.branch_and_bound[:branching_strategy] = branching_strategy

        settings.tolerances[:rel_dual_gap] = 1e-2
        settings.tolerances[:dual_gap] = 1e-6

        settings.smoothing[:generate_smoothing_objective] = generate_smoothing_function
        settings.smoothing[:smoothing_start] = smoothing_start
        settings.smoothing[:smoothing_min] = smoothing_min
        settings.smoothing[:smoothing_min_valid] = smoothing_min_valid
        settings.smoothing[:smoothing_decay] = smoothing_decay
        settings.smoothing[:use_sub_grad_info] = use_sub_grad_info
        settings.smoothing[:max_restart_fw_iter] = min(m,100)

        settings.frank_wolfe[:max_fw_iter] = 5000
        settings.frank_wolfe[:line_search] = line_search
        settings.frank_wolfe[:fw_verbose] = fw_verbose
        settings.frank_wolfe[:lazy_tolerance] = lazy_tolerance
        settings.frank_wolfe[:lazy] = false
        settings.frank_wolfe[:variant] = fw_variant

        settings.tightening[:dual_tightening] = use_tightening
        settings.tightening[:global_dual_tightening] = use_tightening

        settings.heuristic[:hyperplane_aware_rounding_prob] = hyperplane_aware_rounding_prob
        settings.heuristic[:follow_gradient_prob] = follow_gradient_prob
        settings.heuristic[:follow_gradient_steps] = follow_gradient_steps
        settings.heuristic[:rounding_lmo_01_prob] = rounding_lmo_01_prob
        settings.heuristic[:probability_rounding_prob] = probability_rounding_prob
        settings.heuristic[:rounding_prob] = rounding_prob
        settings.heuristic[:custom_heuristics] = custom_heu

        if tree_callback !== nothing
            settings.branch_and_bound[:bnb_callback] = tree_callback
        end
        x, _, result = Boscia.solve(f, sub_grad!, lmo, mode=Boscia.SMOOTHING_MODE, settings=settings)
        
        # Actual run
        @show rounding_prob
        @show N
        settings.branch_and_bound[:verbose] = verbose
        settings.branch_and_bound[:time_limit] = time_limit
        x, _, result = Boscia.solve(f, sub_grad!, lmo, mode=Boscia.SMOOTHING_MODE, settings=settings)
    else
        _, active_set, S = build_start_point2(A, m, n, N, ub)
        line_search = ls_secant ? FrankWolfe.Secant(domain_oracle=domain_oracle) : FrankWolfe.MonotonicGenericStepsize(FrankWolfe.Adaptive(), domain_oracle)
        fw_variant = use_BCG ? Boscia.Blended() : Boscia.BPCG()
        z = greedy_incumbent(A, m, n, N, ub)

        # Precompile
        settings = Boscia.create_default_settings()
        settings.branch_and_bound[:verbose] = false
        settings.branch_and_bound[:time_limit] = 10
        settings.branch_and_bound[:start_solution] = z
        settings.branch_and_bound[:branching_strategy] = branching_strategy
        settings.branch_and_bound[:use_shadow_set] = use_shadow_set
        settings.tightening[:dual_tightening] = use_tightening
        settings.tightening[:global_dual_tightening] = use_tightening
        settings.frank_wolfe[:fw_verbose] = fw_verbose
        settings.frank_wolfe[:lazy_tolerance] = lazy_tolerance
        settings.frank_wolfe[:variant] = fw_variant
        settings.frank_wolfe[:line_search] = line_search
        settings.heuristic[:hyperplane_aware_rounding_prob] = hyperplane_aware_rounding_prob
        settings.heuristic[:follow_gradient_prob] = follow_gradient_prob
        settings.heuristic[:follow_gradient_steps] = follow_gradient_steps
        settings.heuristic[:rounding_lmo_01_prob] = rounding_lmo_01_prob
        settings.heuristic[:probability_rounding_prob] = probability_rounding_prob
        settings.heuristic[:rounding_prob] = rounding_prob
        settings.heuristic[:custom_heuristics] = custom_heu
        settings.domain[:domain_oracle] = domain_oracle
        settings.domain[:active_set] = active_set
        settings.domain[:find_domain_point] = domain_point
        if tree_callback !== nothing
            settings.branch_and_bound[:bnb_callback] = tree_callback
        end
        x, _, result = Boscia.solve(f, grad!, lmo, settings=settings)
        
        _, active_set, S = build_start_point2(A, m, n, N, ub)
        z = greedy_incumbent(A, m, n, N, ub)

        # Actual run
        settings.branch_and_bound[:verbose] = verbose
        settings.branch_and_bound[:time_limit] = time_limit
        settings.branch_and_bound[:start_solution] = z
        settings.domain[:active_set] = active_set
        x, _, result = Boscia.solve(f, grad!, lmo, settings=settings)
    end

    total_time_in_sec=result[:total_time_in_sec]
    status = result[:status_string]
    if occursin("Optimal", result[:status_string])
        status = "OPTIMAL"
    end
    if occursin("Time", result[:status_string])
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
    @show result[:solution_source]

    if write
        #=folder = if long_runs
            "long_runs"
        elseif use_exclusion_criterion
            "exclusion_criterion"
        elseif mu_testing
            string(smoothing_start) * "_" * string(smoothing_decay) * "_" * string(smoothing_min)
        elseif use_heuristics
            "heuristics"
        elseif use_follow_subgradient_heu
            "follow_subgradient"
        elseif use_pipage_heu
            "pipage_rounding"
        elseif use_sr_rounding_heu
            "sr_rounding"
        elseif use_fedorov_heu
            "fedorov"
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
        end =#

        folder = if use_exclusion_criterion
            "exclusion_criterion"
        elseif mu_testing
            string(smoothing_start) * "_" * string(smoothing_decay) * "_" * string(smoothing_min)
        elseif use_heuristics && options_run
            "heuristics"
        elseif use_follow_subgradient_heu && options_run
            "follow_subgradient"
        elseif use_pipage_heu && options_run
            "pipage_rounding"
        elseif use_sr_rounding_heu && options_run
            "sr_rounding"
        elseif use_fedorov_heu && options_run
            "fedorov"
        else
            ""
        end

        @show folder

        connection = criterion == "AGC" ? connected ? "connected" : "disconnected" : ""

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
            file_name = joinpath(@__DIR__, "../csv/full_runs_boscia/boscia_" * folder * "_" * criterion * "_optimality_" * type * "_" * connection * "_" * string(m) * "_" * string(n) * "_" * string(N) * "_" * string(seed) * ".csv")
            CSV.write(file_name, df, append=false)
        end

        idx = findfirst(x -> x == result[:primal_objective], ub_list)
        optimal_time = result[:list_time][idx]
        # CSV file for the results of all instances.
        scaled_solution = result[:primal_objective]*m
        df = DataFrame(
            seed=seed, 
            numberOfExperiments=m, 
            numberOfParameters=n, 
            N=N, 
            time=total_time_in_sec, 
            solution=result[:primal_objective], 
            scaled_solution=scaled_solution, 
            dual_gap = result[:dual_gap],  
            rel_dual_gap=result[:rel_dual_gap], 
            ncalls=result[:lmo_calls], 
            num_nodes=result[:number_nodes],
            termination=status, 
            optimal_time=optimal_time, 
            optimal_iteration=idx, 
            solution_source=String(result[:solution_source]))
        file_name = joinpath(@__DIR__, "../csv/Boscia/boscia_" * folder * "_" * criterion * "_optimality_" * type * "_" * connection * "_" * string(m) * "_" * string(n) * "_" * string(N) * "_" * string(seed) * ".csv" )
        if !isfile(file_name) 
            CSV.write(file_name, df, append=false, writeheader=true, delim=";")
        else 
            CSV.write(file_name, df, append=false, delim=";")
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
