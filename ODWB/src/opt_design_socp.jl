## The SOCP formulation form "COMPUTING EXACT D-OPTIMAL DESIGNS BY MIXED INTEGER SECOND-ORDER CONE PROGRAMMING" 
## by Guillaume Sagnol and Radoslav Harman
"""
Build SOCP of A-Opt.

Following this model (https://picos-api.gitlab.io/picos/optdes.html#exact-a-optimal-design-misocp) instead of the one in the paper.
"""
function build_A_socp_model(seed, m, n, criterion, time_limit, corr, verbose; lb=nothing, ub=nothing, zero_one=false, N=-Inf)
    fusion = false
    p = 0
    if criterion == "AF" 
        A_hat, C, N, ubounds, C_hat = build_data(seed, m, n, true, corr, zero_one=zero_one, N=N)
        fusion = true
        p, _ = size(C_hat)
        A = vcat(C_hat, A_hat)
    else
        A, _, N, ubounds, _ = build_data(seed, m, n, false, corr, zero_one=zero_one, N=N)
        @assert N ≥ n
    end
    @show m, n, N, sum(ubounds) 
    @assert (m > n) && (sum(ubounds) >= N)

    if lb === nothing
        lb = fusion ? vcat(fill(1.0, p), fill(0.0, m)) : fill(0.0, m)
    end
    if ub === nothing
        ub = fusion ? vcat(fill(1.0, p), ubounds) : ubounds
    end

    oa_solver = optimizer_with_attributes(SCIP.Optimizer,
    MOI.Silent() => true, 
    "limits/maxorigsol" => m*10000,
    )
    # SDP solver
    conic_solver = optimizer_with_attributes(Hypatia.Optimizer, 
        MOI.Silent() => !verbose,
    )
    opt = optimizer_with_attributes(Pajarito.Optimizer,
        "time_limit" => time_limit, 
        "iteration_limit" => 1000000,
        "oa_solver" => oa_solver, 
        "conic_solver" => conic_solver,
        MOI.Silent() => !verbose,
    )

    model = Model(opt)
    # add design variables

    JuMP.@variable(model, x[1:(p+m)], Int)
    # Add the probability simplex constraint
    JuMP.@constraint(model, sum(x[(p+1):(p+m)]) == N)
    # Bound constraints
    JuMP.@constraint(model, x ≤ ub)
    JuMP.@constraint(model, x ≥ lb)

    # auxiliary variables 
    # μ 
    JuMP.@variable(model, μ[1:(p+m)])
    # Y_i collected in matrix
    JuMP.@variable(model, Y_mat[1:(p+m), 1:n])

    # additional constraints
    # linear constraints
    JuMP.@constraint(model, sum(A[i, :] * Y_mat[i,:]' for i in 1:(p+m)) .== I)
    # conic constraint
    # second order cone
    for i in 1:(p+m)
        JuMP.@constraint(model, vcat(μ[i] + x[i], vcat(2*Y_mat[i,:], μ[i] - x[i])) in Hypatia.EpiNormEuclCone{Float64}(n+2))
    end

    # objective function
    JuMP.@objective(model, Min, sum(μ))

    return model, x[(p+1):(p+m)]
end

"""
Build the SOCP of the D-criterion
"""
function build_D_socp_model(seed, m, n, criterion, time_limit, corr, verbose; lb=nothing, ub=nothing, zero_one=false, N=-Inf)
    fusion = false
    p = 0
    if criterion == "DF" 
        A_hat, C, N, ubounds, C_hat = build_data(seed, m, n, true, corr, zero_one=zero_one, N=N)
        fusion = true
        p, _ = size(C_hat)
        A = vcat(C_hat, A_hat)
    else
        A, _, N, ubounds, _ = build_data(seed, m, n, false, corr, zero_one, zero_one, N=N)
        @assert N ≥ n
    end
    @show m, n, N, sum(ubounds) 
    @assert (m > n) && (sum(ubounds) >= N)

    if lb === nothing
        lb = fusion ? vcat(fill(1.0, p), fill(0.0, m)) : fill(0.0, m)
    end
    if ub === nothing
        ub = fusion ? vcat(fill(1.0, p), ubounds) : ubounds
    end


    oa_solver = optimizer_with_attributes(SCIP.Optimizer,
    MOI.Silent() => true, 
    "limits/maxorigsol" => m*10000,
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
    # add design variables
    JuMP.@variable(model, w[1:(p+m)])
    JuMP.@variable(model, x[1:(p+m)], Int)
    # Add the probability simplex constraint
    JuMP.@constraint(model, sum(w) == 1)
   # JuMP.@constraint(model, sum(x[(p+1):(p+m)]) == N )
    # Bound constraints
    JuMP.@constraint(model, x ≤ ub)
    JuMP.@constraint(model, x ≥ lb)
    JuMP.@constraint(model, w ≥ fill(0.0, p+m))

    # auxiliary variables 
    # J 
    JuMP.@variable(model, J_mat[1:n, 1:n])
    for i in 1:n 
        if i != n
            for j in (i+1):n
                JuMP.fix(J_mat[i,j], 0; force=true)
            end
        end
    end
    # Z_i collect in a matrix
    JuMP.@variable(model, Z_mat[1:(p+m), 1:n])
    # t_ij collected in a matrix
    JuMP.@variable(model, T_mat[1:(p+m), 1:n])
    # Epigraph variable
    JuMP.@variable(model, t)

    # additional constraints
    # linear constraints
    JuMP.@constraint(model, sum(A[i, :] * Z_mat[i,:]' for i in 1:(p+m)) .- J_mat .== zeros(n,n))
    for j in 1:n
        JuMP.@constraint(model, sum(T_mat[i,j] for i in 1:(p+m)) ≤ J_mat[j,j])
    end
    JuMP.@constraint(model, T_mat .≥ zeros((p+m),n))
    JuMP.@constraint(model, x .== (N+p)*w)
    # conic constraint
    # second order cone
    JuMP.@constraint(model, vcat(t, LinearAlgebra.diag(J_mat)) in Hypatia.HypoGeoMeanCone{Float64}(n+1))
    for i in 1:(p+m)
        for j in 1:n 
            JuMP.@constraint(model, vcat(1/2*T_mat[i,j], w[i], Z_mat[i,j]) in Hypatia.EpiPerSquareCone{Float64}(3))
        end
    end

    # objective function
    JuMP.@objective(model, Max, t)

    return model, x[(p+1):(p+m)]
end

function solve_opt_socp(seed, m, n, time_limit, criterion, corr; write=true, verbose=true, zero_one=false, N=-Inf)
    if criterion == "DF" || criterion == "D"
        model, x = build_D_socp_model(seed, m, n, criterion, 10, corr, false, N=N)
        optimize!(model)
        model, x = build_D_socp_model(seed, m, n, criterion, time_limit, corr, verbose, zero_one=zero_one)
    elseif criterion == "AF" || criterion == "A"
        model, x= build_A_socp_model(seed, m, n, criterion, 10, corr, false, N=N)
        optimize!(model)
        model, x = build_A_socp_model(seed, m, n, criterion, time_limit, corr, verbose, zero_one=zero_one)
    end

    # solve 
    optimize!(model)

    # query solution
    status = termination_status(model)
    solution = objective_value(model)
    y = value.(x)
    t = solve_time(model)
    paja_opt = JuMP.unsafe_backend(model)
    numberIter = paja_opt.num_iters
    numberCuts = paja_opt.num_cuts

    # Check feasibility
    if criterion == "A" || criterion == "D"
        A, C, N, ub, _ = build_data(seed, m, n, false, corr, N=N)
    elseif criterion == "AF"|| criterion == "DF"
        A, C, N, ub, _ = build_data(seed, m, n, true, corr, N=N)
    end
    if criterion in ["A","AF"]
        f_check, _ = build_a_criterion(A, criterion == "AF", C=C, build_safe = false, μ=criterion == "A" ? 1e-4 : 0.0)
    elseif criterion in ["GTI","GTIF"]
        f_check, _ = build_general_trace(A, p, criterion == "GTIF", C=C)
    else
        f_check, _ = build_d_criterion(A, criterion == "DF", C=C, build_safe = false, μ=criterion == "D" ? 1e-4 : 0.0)
    end
    feasible = isfeasible(seed, m, n,criterion, y, corr)
    @show feasible
    scaled_solution = if feasible
        f_check(y)
    else
        Inf
    end

    type = corr ? "correlated" : "independent"
    @show status
    @show y
    @show solution
    @show scaled_solution

    if write 
        df = DataFrame(seed=seed, numberOfExperiments=m, numberOfParameters=n, time=t, N=N, solution=solution, scaled_solution=scaled_solution, termination=status, numberIterations=numberIter, numberCuts=numberCuts, feasible = feasible)
        file_name = joinpath(@__DIR__, "../csv/SOCP/socp_" * criterion * "_optimality_" * type * "_" * string(m) * "_" * string(n) * "_" * string(seed) * ".csv")
        CSV.write(file_name, df, append=false)
    end
    return y
end
