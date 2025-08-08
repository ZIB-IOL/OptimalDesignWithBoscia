"""
    Custom Branch-and-Bound for the Exact Optimal Design Problem
"""

"""
Node struct for the tree. Knows its bounds, the solution vector, number of iterations and time taken.
"""
mutable struct CustomBBNode <: Bonobo.AbstractNode
    std::Bonobo.BnBNodeInfo
    lower_bounds::Vector{Float64}
    upper_bounds::Vector{Float64}
    solution::Vector{Float64}
    w::Vector{Float64}
    time::Float64
    iteration_count::Int64
end

mutable struct ConstrainedBoxMIProblem{F,G,L,D}
    f::F
    grad::G
    linesearch::L
    domain_oralce::D
    nvars::Int64
    integer_vars::Vector{Int64}
    N::Float64
    time_limit::Union{Int64, Float64}
    rel_gap::Float64
    abs_gap::Float64
    Stage::Boscia.Solve_Stage
end

ConstrainedBoxMIProblem(f, g, linesearch, domain_oracle, n, int_vars, N, time_limit) =
    ConstrainedBoxMIProblem(f, g, linesearch, domain_oracle, n, int_vars, N, time_limit, 1e-2, 1e-6, Boscia.SOLVING)

    """
Returns the solution vector of the relaxed node problem.
"""
function Bonobo.get_relaxed_values(tree::Bonobo.BnBTree{<:CustomBBNode}, node)
    return node.solution
end

"""
Create the information for the childern nodes.
"""
function Bonobo.get_branching_nodes_info(tree::Bonobo.BnBTree{<:CustomBBNode}, node, vidx)
    x = Bonobo.get_relaxed_values(tree, node)
    w = node.w
    N = tree.root.N
    @assert node.lower_bounds[vidx] < node.upper_bounds[vidx] "Branching lb: $(node.lower_bounds[vidx]) ub: $(node.upper_bounds[vidx]) x[vidx]=$(x[vidx])"
    frac_val = x[vidx]

    #@show frac_val, w[vidx], node.upper_bounds[vidx], node.lower_bounds[vidx]

    # Left child keeps the lower bounds and gets a new upper bound at vidx.
    left_bounds = copy(node.upper_bounds)
    left_bounds[vidx] = floor(frac_val)/N
    left_solution = copy(node.solution)
    left_solution[vidx] = floor(frac_val)
    left_w = copy(node.w)
    left_w[vidx] = floor(frac_val)/N


    node_info_left = (lower_bounds=node.lower_bounds, 
                    upper_bounds = left_bounds,
                    solution = left_solution,
                    w = left_w,
                    time = 0.0,
                    iteration_count = 0
    )

    # Right child keeps the upper bounds and gets a new lower bound at vidx.
    right_bounds = copy(node.lower_bounds)
    right_bounds[vidx] = ceil(frac_val)/N
    right_solution = copy(node.solution)
    right_solution[vidx] = ceil(frac_val)
    right_w = copy(node.w)
    right_w[vidx] = ceil(frac_val)/N

    node_info_right = (lower_bounds=right_bounds, 
                    upper_bounds = node.upper_bounds,
                    solution = right_solution,
                    w = right_w,
                    time = 0.0,
                    iteration_count = 0
    )

    return node_info_left, node_info_right
end

"""
Solve the box constrained approximate optimal design problem.
"""
function Bonobo.evaluate_node!(tree, node)
    # count time and iterations
    time_ref = Dates.now()
    iter_count = 0

    # get feasible start point
    N = tree.root.N
    w = copy(node.w)
    w = alternating_projection(w, node, 1)

    # If the node is not reachable, i.e. we can never satisfy the knapsack add_constraint
    if w === nothing
        return NaN, NaN
    end
    x = N*w

    #@assert check_feasibilty(w, node, tree)

    gradient = similar(w)
    tree.root.grad(gradient, w)
    linesearch_workspace = FrankWolfe.build_linesearch_workspace(FrankWolfe.Adaptive(), w, gradient)

    jdx = find_max_gradient(w, -gradient, node.upper_bounds, tree.root)
    kdx = find_min_gradient(w, -gradient, node.lower_bounds, tree.root)

    while gradient[jdx]/gradient[kdx] - 1 > 1e-6 && iter_count < 10000
        # find theta
        d = zeros(length(w))
        d[jdx] = 1.0
        d[kdx] = -1.0
        theta_max = min(node.upper_bounds[jdx] - w[jdx], w[kdx] - node.lower_bounds[kdx])
        theta =  tree.root.linesearch(
            node,
            tree.root.f,
            tree.root.grad,
            gradient, 
            w,
            d,
            theta_max,
            linesearch_workspace,
            iter_count,
            jdx,
            kdx
        )

        # weight shift
        w[jdx] += theta
        w[kdx] -= theta

        # compute the new jdx and kdx
        tree.root.grad(gradient, w)
        jdx = find_max_gradient(w, -gradient, node.upper_bounds, tree.root)
        kdx = find_min_gradient(w, -gradient, node.lower_bounds, tree.root)
        iter_count += 1

        if !check_feasibilty(w, node)
            @show node.upper_bounds - w
            @show w - node.lower_bounds
            @show sum(w)
        end
        @assert check_feasibilty(w, node)
    end

    time = float(Dates.value(Dates.now() - time_ref))
    node.time = time
    node.iteration_count = iter_count

    x = N*w
    node.w = w
    node.solution = x
    obj_value = tree.root.f(x)

    if isapprox(sum(isapprox.(x, round.(x); atol=1e-6, rtol=5e-2)), tree.root.nvars)  && check_feasibilty(round.(x),node.lower_bounds*N, node.upper_bounds*N, N)
        println("Integer solution found.")
        node.solution = round.(x)
        primal = tree.root.f(node.solution)
        return obj_value, primal
    end

    rounding_heuristics(tree, node, x)

    return obj_value, NaN
end

"""
    Alternating projection

See: https://en.wikipedia.org/wiki/Projections_onto_convex_sets
"""
function alternating_projection(x, node, N)
    if sum(node.lower_bounds) > N || sum(node.upper_bounds) < N || !all(node.lower_bounds <= node.upper_bounds)
        return nothing
    end

    function project_onto_cube(x, lb, ub)
        return clamp.(x, lb, ub)
    end
    function project_on_prob_simplex(x, N)
        n = length(x)
        return x - (sum(x)-N)/n * ones(n)
    end

    iter = 0

    while !check_feasibilty(x, node.lower_bounds, node.upper_bounds, N) && iter < 100000
        x = project_onto_cube(x, node.lower_bounds, node.upper_bounds)
        x = project_on_prob_simplex(x, N) 
        iter += 1
    end
    @debug "Number of iteration in the alternating projection: $(iter)"

    return x
end

"""
Simply rounding heuristic.
"""
function rounding_heuristics(tree, node, x)
    y = copy(x)
    y = round.(y)
    if count(!iszero, y) == 0
        return 
    end
    N = tree.root.N

    non_zero_int = findall(!iszero, y)
    if sum(node.upper_bounds) < N || sum(node.lower_bounds) > N
        @debug "No heuristics improvement possible, bounds already reached, N=$(N), maximal possible sum $(sum(node.upper_bounds)), minimal possible sum $(sum(node.lower_bounds))"
        return
    end

    if sum(y) < N
        while sum(y) < N
            y = add_to_min2(y, node.upper_bounds)
        end
    elseif sum(y) > N
        while sum(y) > N
            if iszero.(y) + isone.(y) == ones(tree.root.nvars) 
                return
            end
            y = remove_from_max(y)
        end
    end

    if !check_feasibilty(y, node, N=N) || !domain_oracle(y)
        return 
    end

    fy = tree.root.f(y)
    sol = Bonobo.DefaultSolution(fy, y, node)
    push!(tree.solutions, sol)

    push!(tree.solutions, sol)
    if tree.incumbent_solution === nothing ||fy < tree.incumbent_solution.objective
        tree.incumbent_solution = sol
        tree.incumbent = fy
    end
end


"""
Check feasibility of a given point.
"""
function check_feasibilty(x, lb, ub, N; tol=0.0)
    m = length(x)
    if all(x.>=lb .- tol) && all(x.<=ub .+ tol) && isapprox(sum(x), N) 
        return true
    else
        @debug "Sum of w $(sum(x))"
        @debug "Lower bounds: $(lb)"
        @debug "Iterate: $(x)"
        @debug "Upper bounds: $(ub)"
        @debug "Difference iterate and lower bounds: $(x-lb)"
        @debug "Difference iterate and upper bounds: $(ub-x)"
        return false
    end
    return sum(x.>=lb) == m && sum(x.<=ub) == m && isapprox(sum(x), N) 
end
check_feasibilty(x, node; N=1, tol=1e-9) = check_feasibilty(x, node.lower_bounds, node.upper_bounds, N, tol=tol)

"""
Returns the list of indices corresponding to the integral variables.
"""
function Bonobo.get_branching_indices(root::ConstrainedBoxMIProblem)
    return root.integer_vars
end

"""
Find maximum gradient entry where the upper bound has yet been met.
"""
function find_max_gradient(x, gradient, ub, root; tol=0.0)
    int_var = Bonobo.get_branching_indices(root)
    bool_vec = x[int_var] .< ub[int_var] .+ tol
    idx_set = findall(x -> x == 1, bool_vec) 
    return findfirst(x -> x == maximum(gradient[idx_set]), gradient)
end

"""
Find minimum gradient entry where the lower bound has yet been met.
"""
function find_min_gradient(x, gradient, lb, root; tol=0.0)
    int_var = Bonobo.get_branching_indices(root)
    bool_vec = x[int_var] .> lb[int_var] .- tol
    idx_set = findall(x -> x == 1, bool_vec) 
    return findfirst(x -> x == minimum(gradient[idx_set]), gradient)
end

"""
Set up the problem and solve it with the custom BB algorithm.
"""
function solve_opt_custom(seed, m, n, time_limit, criterion, corr; p=0, write = true, verbose= true, long_run=false, print_iter=100, specific_seed=false)
    # build data
    A, C, N, ub, C_hat = build_data(seed, m, n, criterion in ["AF","DF", "GTIF"], corr, scaling_C=long_run)
    # build function
    if criterion == "A" || criterion == "AF"
        p = -1
    elseif criterion == "D" || criterion == "DF"
        p = 0
    end
    @assert p <= 0
    if criterion in ["A","D","GTI"]
        C_hat = nothing
    end
    build_safe = criterion in ["A","D","GTI"]
    @show build_safe
    if criterion == "A" && ((m==60 && n==15 && seed == 4 && corr) || (m==100 && n==30 && seed==1 && corr) || (m==50 && n==12 && seed == 5 && !corr))
        μ = 1e-4
        @show μ
    else
        μ = 0.0
    end
    f, grad!, linesearch = build_matrix_means_objective(A, p, μ=μ, C_hat=C_hat, build_safe = build_safe)
    domain_oracle = build_domain_oracle(A, n)

    # create problem and tree
    if criterion in ["A","D", "GTI"]
        x0 = greedy_incumbent(A, m, n, N, ub)
        m_hat = m
        ub_hat = ub
        N_hat = N
        lb_hat = zeros(m)
    else
        x0 = vcat(greedy_incumbent_fusion(A,m,n,N,ub), fill(1, 2n))
        m_hat = m + 2n
        ub_hat = vcat(ub, ones(2n))
        N_hat = N + 2n
        lb_hat = vcat(zeros(m), ones(2n)) 
    end
    root = ConstrainedBoxMIProblem(f, grad!, linesearch, domain_oracle, m_hat, collect(1:m), N_hat, time_limit)
    nodeExample = CustomBBNode(
        Bonobo.BnBNodeInfo(1, 0.0, 0.0),
        lb_hat,
        ub_hat, 
        fill(-1.0, length(x0)),
        fill(-1.0, length(x0)),
        0.0,
        0
    )

    Node = typeof(nodeExample)
    Value = Vector{Float64}
    tree = Bonobo.initialize(;
        traverse_strategy = Bonobo.BFS(),
        branch_strategy = Bonobo.MOST_INFEASIBLE(),
        Node = Node,
        Solution = Bonobo.DefaultSolution{Node, Value},
        root = root,
        sense = :Min
    )

    Bonobo.set_root!(tree, (
        lower_bounds = lb_hat/N_hat,
        upper_bounds = ub_hat/N_hat,
        solution = x0,
        w = x0/N_hat,
        time=0.0,
        iteration_count=0
    ))

    time_ref = Dates.now()
    callback = build_callback(tree, time_ref, verbose, print_iter)

    # dummy solution - In case the process stops because of the time limit and no solution as been found yet.
    #x0, _ = optDesign.solve_opt(seed, m, n, 300, criterion, corr, write=false, verbose =false)
    dummy_solution = Bonobo.DefaultSolution(Inf, fill(-1.0, length(x0)), nodeExample)
    if isapprox(sum(isapprox.(x0, round.(x0))), tree.root.nvars)
        # Minus f(x0) because the tree internally treats this as a min problem
        first_solution = Bonobo.DefaultSolution(tree.root.f(x0), x0, nodeExample)
        push!(tree.solutions, first_solution)
        tree.incumbent_solution = first_solution
        tree.incumbent = tree.root.f(x0)
    else
        push!(tree.solutions, dummy_solution)
        tree.incumbent_solution = dummy_solution
    end

    # Optimize
    sol_data = @timed Bonobo.optimize!(tree,callback=callback)

    time = sol_data.time
    # post processing 
    sol_x = convert.(Int, Bonobo.get_solution(tree))
    if tree.root.Stage == Boscia.OPT_TREE_EMPTY
        @assert sum(sol_x .!= dummy_solution.solution) == length(sol_x)
    end
    @show sol_x[1:m]
    solution = f(sol_x)
    @show solution
    @show tree.num_nodes

    # check feasibility
    feasible = isfeasible(seed, m, n, criterion, sol_x[1:m], corr, N=N)
    if !feasible
        @show m, sum(ub - sol_x[1:m] .>= 0)
        @show m, sum(sol_x[1:m] .>= 0)
        @show N, sum(sol_x[1:m])
    end
    @assert feasible

    # Calculate objective with respect to the criteria used in Boscia
    if criterion in ["A","AF"]
        f_check, _ = build_a_criterion(A, criterion == "AF", C=C, build_safe = false, μ=criterion == "A" ? 1e-4 : 0.0, long_run=long_run)
    elseif criterion in ["GTI","GTIF"]
        f_check, _ = build_general_trace(A, -p, criterion == "GTIF", C=C)
    else
        f_check, _ = build_d_criterion(A, criterion == "DF", C=C, build_safe = false, μ=criterion == "D" ? 1e-4 : 0.0, long_run=long_run)
    end
    solution_scaled = f_check(sol_x[1:m])
    #solution_scaled = criterion == "A" ? exp(solution) : solution - n*log(n)
    @show solution_scaled

    # Check the solving stage
    @assert tree.root.Stage != Boscia.SOLVING
    @show tree.root.Stage
    status = if tree.root.Stage in [Boscia.OPT_GAP_REACHED, Boscia.OPT_TREE_EMPTY]
        "optimal"
    else
        "TIME_LIMIT"
    end

    dual_gap = tree.incumbent - tree.lb
    @show dual_gap

    if write 
        subfolder = if criterion in ["GTI", "GTIF"]
            "GTI"
        elseif long_run
            "long_runs"
        else
            ""
        end
        if criterion in ["GTI","GTIF"]
            criterion = criterion * "_" * string(Int64((-1) * p *100))
        end
        # write data into file
        time_per_nodes = time/tree.num_nodes
        type = corr ? "correlated" : "independent"
        df = DataFrame(seed=seed, numberOfExperiments=m, numberOfParameters=n, N=N, time=time, time_per_nodes=time_per_nodes, solution=solution, solution_scaled=solution_scaled, dual_gap=dual_gap, number_nodes=tree.num_nodes, termination=status)
        file_name = joinpath(@__DIR__, "../csv/Co-BnB/" * subfolder * "/co-bnb_" * criterion * "_optimality_" * type * "_" * string(m) * "_" * string(n) * "_" * string(seed) * ".csv")

        CSV.write(file_name, df, append=false)
        #if !isfile(file_name)
        #    CSV.write(file_name, df, append=true, writeheader=true)
        #else 
        #    CSV.write(file_name, df, append=true)
        #end
    end
    return sol_x
end

function build_callback(tree, time_ref, verbose, print_iter)
    iter = 1
    return function callback(tree, node; node_infeasible = false, worse_than_incumbent = false)

        time = float(Dates.value(Dates.now() - time_ref))
        if (node.id == 1 || mod(iter, print_iter) == 0  || Bonobo.terminated(tree)) && verbose
            println("Iter: $(iter) ID: $(node.id) num_nodes: $(tree.num_nodes) LB: $(tree.lb) Incumbent: $(tree.incumbent) Total Time in s: $(time/1000.0) Time node in s: $(node.time/1000.0) Iterations: $(node.iteration_count)")
            if node_infeasible
                println("Node not feasible!")
            elseif worse_than_incumbent
                println("Node cut because it is worse than the incumbent")
            end
        end
        iter += 1

        # break if time is met
        if tree.root.time_limit < Inf
            if time / 1000.0 ≥ tree.root.time_limit
                if tree.root.Stage == Boscia.SOLVING
                    @assert tree.root.Stage == Boscia.SOLVING
                    tree.root.Stage = Boscia.TIME_LIMIT_REACHED
                end
            end
        end
    end
end

"""
Checks if the branch and bound can be stopped.
By default (in Bonobo) stops then the priority queue is empty. 
"""
function Bonobo.terminated(tree::Bonobo.BnBTree{<:CustomBBNode})
    # time limit reached
    if tree.root.Stage == Boscia.TIME_LIMIT_REACHED
        return true
    end

    # absolute gap reached
    absgap = tree.incumbent - tree.lb
    if absgap ≤ tree.root.abs_gap
        tree.root.Stage = Boscia.OPT_GAP_REACHED
        @show absgap
        return true
    end

    # relative gap reached
    dual_gap = if signbit(tree.incumbent) != signbit(tree.lb)
        Inf
    elseif tree.incumbent == tree.lb
        0.0
    else
        absgap / min(abs(tree.incumbent), abs(tree.lb))
    end
    if dual_gap ≤ tree.root.rel_gap
        tree.root.Stage = Boscia.OPT_GAP_REACHED
        @show dual_gap
        return true
    end

    # tree empty
    if isempty(tree.nodes)
        tree.root.Stage = Boscia.OPT_TREE_EMPTY
        return true
    end

    return false
end

"""
Own optimize function.
"""
function Bonobo.optimize!(tree::Bonobo.BnBTree{<:CustomBBNode}; callback=(args...; kwargs...)->())
    while !Bonobo.terminated(tree)
        node = Bonobo.get_next_node(tree, tree.options.traverse_strategy)
        lb, ub = Bonobo.evaluate_node!(tree, node)
        # if the problem was infeasible we simply close the node and continue
        if isnan(lb) && isnan(ub)
            Bonobo.close_node!(tree, node)
            callback(tree, node; node_infeasible=true)
            continue
        end

        Bonobo.set_node_bound!(tree.sense, node, lb, ub)
       
        # if the evaluated lower bound is worse than the best incumbent -> close and continue
        if node.lb >= tree.incumbent
            Bonobo.close_node!(tree, node)
            callback(tree, node; worse_than_incumbent=true)
            continue
        end

        tree.node_queue[node.id] = (node.lb, node.id)
        _ , prio = peek(tree.node_queue)
        @assert tree.lb <= prio[1]
        tree.lb = prio[1]

        updated = Bonobo.update_best_solution!(tree, node)
        if updated
            Bonobo.bound!(tree, node.id)
            if isapprox(tree.incumbent, tree.lb; atol=tree.options.atol, rtol=tree.options.rtol)
                tree.root.Stage = Boscia.OPT_GAP_REACHED
                callback(tree, node)
                break
            end
        end

        Bonobo.close_node!(tree, node)
        Bonobo.branch!(tree, node)
        callback(tree, node)
    end
    Bonobo.sort_solutions!(tree.solutions, tree.sense)
end