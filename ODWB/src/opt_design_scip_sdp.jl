# SCIP SDP: Use SCIP-SDP when possible via CBF round-trip (avoids checkVarsLocks assertion).
# Fallback: Pajarito (HiGHS+Hypatia) when use_scip_sdp=false.

function _build_eopt_model_for_cbf(seed, m, n, criterion, corr; integer_data=false, zero_one=false, N=-Inf)
    if criterion == "EF"
        A, C, N, ub, _ = integer_data ? build_integer_data(seed, m, n, true, corr, zero_one=zero_one, N=N) : build_data(seed, m, n, true, corr, zero_one=zero_one, N=N)
    else
        A, _, N, ub, _ = integer_data ? build_integer_data(seed, m, n, false, corr, zero_one=zero_one, N=N) : build_data(seed, m, n, false, corr, zero_one=zero_one, N=N)
        C = nothing
    end
    model = Model()
    @variable(model, x[1:m])
    JuMP.set_integer.(x)
    @variable(model, t)
    @constraint(model, sum(x) == N)
    @constraint(model, x >= 0)
    @constraint(model, x <= ub)
    if criterion == "E"
        info_matrix = [
            @expression(model, (i == j ? -t : 0.0) + sum(A[k, i] * x[k] * A[k, j] for k in 1:m))
            for i in 1:n, j in 1:n
        ]
        @constraint(model, info_matrix in JuMP.PSDCone())
    else
        info_matrix = [
            @expression(model, C[i, j] + (i == j ? -t : 0.0) + sum(A[k, i] * x[k] * A[k, j] for k in 1:m))
            for i in 1:n, j in 1:n
        ]
        @constraint(model, info_matrix in JuMP.PSDCone())
    end
    @objective(model, Max, t)
    return model, x, t, m
end

function _export_model_to_cbf(model, filename)
    cbf_model = MOI.FileFormats.Model(format=MOI.FileFormats.FORMAT_CBF)
    bridged = MOI.Bridges.full_bridge_optimizer(cbf_model, Float64)
    MOI.copy_to(bridged, backend(model))
    MOI.write_to_file(cbf_model, filename)
end

#=function build_E_scipsdp_model(seed, m, n, criterion, time_limit, corr; verbose=true, integer_data=false, zero_one=false, N=-Inf, use_scip_sdp=false)
    if criterion == "EF"
        A, C, N, ub, _ = integer_data ? build_integer_data(seed, m, n, true, corr, zero_one=zero_one, N=N) : build_data(seed, m, n, true, corr, zero_one=zero_one, N=N)
    else
        A, _, N, ub, _ = integer_data ? build_integer_data(seed, m, n, false, corr, zero_one=zero_one, N=N) : build_data(seed, m, n, false, corr, zero_one=zero_one, N=N)
    end

    @eval(SCIP, have_scip_sdp) || error("SCIP-SDP required. Set SCIP_SDP_OPTDIR and rebuild SCIP.")
    opt = optimizer_with_attributes(SCIP.Optimizer,
        MOI.Silent() => !verbose,
        "limits/time" => time_limit,
        "limits/gap" => 1e-6,
        MOI.RawOptimizerAttribute("relaxing/SDP/freq") => -1,
        MOI.RawOptimizerAttribute("lp/solvefreq") => 1,
    )
    model = Model(opt)

    JuMP.@variable(model, x[1:m])
    JuMP.set_integer.(x)
    JuMP.@variable(model, t)

    if use_scip_sdp
        # SCIP-SDP: add PSD constraint first (order may affect lock handling)
        if criterion == "E"
            info_matrix = [
                (i == j ? -t : 0.0) + sum(A[k, i] * x[k] * A[k, j] for k in 1:m)
                for i in 1:n, j in 1:n
            ]
        else
            info_matrix = [
                C[i, j] + (i == j ? -t : 0.0) + sum(A[k, i] * x[k] * A[k, j] for k in 1:m)
                for i in 1:n, j in 1:n
            ]
        end
        JuMP.@constraint(model, info_matrix in JuMP.PSDCone())
    else
        if criterion == "E"
            info_matrix = [
                JuMP.@expression(model, (i == j ? -t : 0.0) + sum(A[k, i] * x[k] * A[k, j] for k in 1:m))
                for i in 1:n, j in 1:n
            ]
        else
            info_matrix = [
                JuMP.@expression(model, C[i, j] + (i == j ? -t : 0.0) + sum(A[k, i] * x[k] * A[k, j] for k in 1:m))
                for i in 1:n, j in 1:n
            ]
        end
        JuMP.@constraint(model, info_matrix in JuMP.PSDCone())
    end

    JuMP.@constraint(model, sum(x) == N)
    JuMP.@constraint(model, x >= 0)
    JuMP.@constraint(model, x <= ub)
    @objective(model, Max, t)
    

    return model, x, t
end =#

const _SCIP_STATUS_TO_MOI = Dict(
    SCIP.SCIP_STATUS_OPTIMAL => MOI.OPTIMAL,
    SCIP.SCIP_STATUS_INFEASIBLE => MOI.INFEASIBLE,
    SCIP.SCIP_STATUS_UNBOUNDED => MOI.DUAL_INFEASIBLE,
    SCIP.SCIP_STATUS_TIMELIMIT => MOI.TIME_LIMIT,
    SCIP.SCIP_STATUS_NODELIMIT => MOI.NODE_LIMIT,
    SCIP.SCIP_STATUS_GAPLIMIT => MOI.OPTIMAL,
)

function solve_opt_scip_sdp(
    seed, 
    m, 
    n, 
    time_limit, 
    criterion, 
    corr; 
    write=true, 
    verbose=true, 
    integer_data=false, 
    boscia_solution=nothing, 
    zero_one=false, 
    N=-Inf, 
    use_scip_sdp=true, 
    scip_sdp_mode=:oa, 
    return_diagnostics=false
    )
    if criterion != "E" && criterion != "EF"
        error("SCIP SDP can currently only handle E-optimal and EF-optimal problems")
    end

    @assert SCIP.have_scip_sdp "SCIP-SDP required. Set SCIP_SDP_OPTDIR and rebuild SCIP."
    # CBF round-trip: avoids checkVarsLocks assertion when vars appear in SDP + linear constraints
    model, x_ref, t_ref, m_dim = _build_eopt_model_for_cbf(seed, m, n, criterion, corr; integer_data, zero_one, N)
    cbf_path = joinpath(mktempdir(), "eopt_scip_sdp_$(getpid()).cbf")
    
    _export_model_to_cbf(model, cbf_path)
    # Precompile: 10s run to trigger JIT and avoid large first-run compile (same pattern as other solvers)
    SCIP.solve_cbf_with_scip_sdp(cbf_path; time_limit=10, gap=1e-2, verbose=false, sdp_mode=scip_sdp_mode)
    # Actual run
    result = SCIP.solve_cbf_with_scip_sdp(cbf_path; time_limit, gap=1e-2, verbose, sdp_mode=scip_sdp_mode)
    status = get(_SCIP_STATUS_TO_MOI, result.status, MOI.OTHER_ERROR)
    solution = result.obj_val
    t = result.solve_time
    # CBF order: x[1..m] then t (m+1 vars total)
    ord = result.var_values_ordered
    y = length(ord) >= m ? ord[1:m] : fill(NaN, m)
    isfile(cbf_path) && rm(cbf_path, force=true)
    @show y

    # Diagnostics: n_nodes (both modes), n_cuts_found/n_cuts_applied (OA), n_sdp_iters (B&B; nothing if not exposed)
    diagnostics = (; n_nodes=result.n_nodes, n_cuts_found=result.n_cuts_found, n_cuts_applied=result.n_cuts_applied, n_sdp_iters=result.n_sdp_iters)

    @show diagnostics

    A, C, N, ub, _ = integer_data ? build_integer_data(seed, m, n, false, corr, zero_one=zero_one, N=N) : build_data(seed, m, n, false, corr, zero_one=zero_one, N=N)
    f_check, _ = build_e_criterion(A)
    feasible = isfeasible(seed, m, n, criterion, y, corr, ub=ub)
    scaled_solution = feasible ? f_check(y) : Inf
    @show feasible, scaled_solution

    if boscia_solution !== nothing
        @show boscia_solution
        @show isfeasible(seed, m, n, criterion, boscia_solution, corr, ub=ub)
        @show f_check(boscia_solution)
        @show abs(f_check(boscia_solution) - f_check(y))/min(abs(f_check(boscia_solution)), abs(f_check(y)))
    end

    if write
        integer_data_str = integer_data ? "_int_" : "_cont_"
        mode = use_scip_sdp ? "SCIPSDP" : "Pajarito"
        sdp_mode_str = use_scip_sdp ? "_$(scip_sdp_mode)" : ""
        df = DataFrame(
            seed=seed, numberOfExperiments=m, numberOfParameters=n, time=t, N=N,
            solution=solution, scaled_solution=scaled_solution, termination=status, feasible=feasible,
            n_nodes=diagnostics.n_nodes,
            n_cuts_found=diagnostics.n_cuts_found, n_cuts_applied=diagnostics.n_cuts_applied,
            n_sdp_iters=something(diagnostics.n_sdp_iters, missing),
        )
        file_name = joinpath(@__DIR__, "../csv/SCIPSDP/scip_sdp_$(mode)$(sdp_mode_str)_$(criterion)_optimality_$(corr ? "correlated" : "independent")$(integer_data_str)_$(m)_$(n)_$(N)_$(seed).csv")
        isfile(file_name) ? CSV.write(file_name, df, append=true) : CSV.write(file_name, df, writeheader=true)
    end
    return_diagnostics ? (y, diagnostics) : y
end
