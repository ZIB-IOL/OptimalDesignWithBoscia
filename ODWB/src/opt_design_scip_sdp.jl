# SCIP SDP: Use SCIP-SDP when possible via CBF round-trip (avoids checkVarsLocks assertion).
# Fallback: Pajarito (HiGHS+Hypatia) when use_scip_sdp=false.

function _build_eopt_model_for_cbf(seed, m, n, criterion, corr; zero_one=false, N=-Inf, connected=true, tightened=false, scaled_input=false)
    if criterion == "EF" 
        A, C, N, ub, _ = build_data(seed, m, n, true, corr, zero_one=zero_one, N=N)
    elseif criterion == "AGC"
        present_edges = connected ? Int(floor(2 * m)) : Int(floor(1/2 * m))
        edges, potential_edges = build_graph_connectivity_data(n, present_edges, m, seed=seed, connected=connected)
        L = graph_laplacian(n, edges)
        A = potential_edges_incidence_matrix(n, potential_edges)
        C = L + ones(n, n)
        ub = fill(1.0, m)
        N = !isfinite(N) ? Int(floor(m/2)) : N
    else
        A, _, N, ub, _ = build_data(seed, m, n, false, corr, zero_one=zero_one, N=N)
        C = nothing
    end
    scale = minimum(eigvals(A' * A))
    A = scaled_input ? A / scale : A
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
    if tightened
        @constraint(model, t * (n - N + 1) <= sum(x[i] * norm(A[i, :], 2)^2 for i in 1:m))
    end
    @objective(model, Max, t)
    return model, x, t
end
# Returns JuMP VariableIndex → 1-based column order in the written CBF (matches `var_values_ordered` from SCIP).
function _export_model_to_cbf(model, filename)
    cbf_model = MOI.FileFormats.Model(format=MOI.FileFormats.FORMAT_CBF)
    bridged = MOI.Bridges.full_bridge_optimizer(cbf_model, Float64)
    src = JuMP.backend(model)
    index_map = MOI.copy_to(bridged, src)
    dest_order = MOI.get(bridged, MOI.ListOfVariableIndices())
    dest_to_pos = Dict{MOI.VariableIndex,Int}(zip(dest_order, eachindex(dest_order)))
    jump_vi_to_pos = Dict{MOI.VariableIndex,Int}()
    for vi in MOI.get(src, MOI.ListOfVariableIndices())
        dest_vi = index_map[vi]
        if haskey(dest_to_pos, dest_vi)
            jump_vi_to_pos[vi] = dest_to_pos[dest_vi]
        end
    end
    MOI.write_to_file(cbf_model, filename)
    return jump_vi_to_pos
end

const _SCIP_STATUS_TO_MOI = Dict(
    SCIP.SCIP_STATUS_OPTIMAL => MOI.OPTIMAL,
    SCIP.SCIP_STATUS_INFEASIBLE => MOI.INFEASIBLE,
    SCIP.SCIP_STATUS_UNBOUNDED => MOI.DUAL_INFEASIBLE,
    SCIP.SCIP_STATUS_TIMELIMIT => MOI.TIME_LIMIT,
    SCIP.SCIP_STATUS_NODELIMIT => MOI.NODE_LIMIT,
    SCIP.SCIP_STATUS_GAPLIMIT => MOI.OPTIMAL,
)

# SCIP SDP on mac: /Users/deborah/SCIP-SDP/build

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
    zero_one=true, 
    N=-Inf, 
    scip_sdp_mode=:oa, 
    return_diagnostics=false,
    connected=true,
    tightened=false,
    gap=1e-6,
    rel_gap=1e-2,
    scale=Inf,
    augment_budget=-1,
    use_base_graph=false,
    presolve=true,
    symmetry=true,
    disable_crossover_heuristic=false,
    disable_heuristics=false,
    extra_scip_params=Dict{String,Any}(),
    scaled_input=false,
    )
    if !(criterion in ["E", "EF", "AGC","ACST"])
        error("SCIP SDP can currently only handle E-optimal, EF-optimal, AGC and ACST problems")
    end

    @assert SCIP.have_scip_sdp "SCIP-SDP required. Set SCIP_SDP_OPTDIR and rebuild SCIP."
    # CBF round-trip: avoids checkVarsLocks assertion when vars appear in SDP + linear constraints
    model, x_lin, x_mat = if criterion in ["E", "EF", "AGC"]
        mod, xv, _t = _build_eopt_model_for_cbf(seed, m, n, criterion, corr; zero_one, N, connected, tightened, scaled_input)
        (mod, xv, nothing)
    else
        mod, _γ, xm = algebraic_connectivity_model(
            seed, m, n;
            build_spanning_tree = true,
            use_base_graph = use_base_graph,
            augment_budget = augment_budget,
        )
        (mod, nothing, xm)
    end
    cbf_path = joinpath(mktempdir(), "eopt_scip_sdp_$(getpid()).cbf")
    
    jump_vi_to_pos = _export_model_to_cbf(model, cbf_path)
    # Precompile: 10s run to trigger JIT and avoid large first-run compile (same pattern as other solvers)
    gap = N < n ? 1e-4 : gap
    SCIP.solve_cbf_with_scip_sdp(
        cbf_path;
        time_limit=10,
        gap=rel_gap,
        absgap=gap,
        verbose=false,
        sdp_mode=scip_sdp_mode,
        presolving=presolve,
        symmetry=symmetry,
        disable_crossover_heuristic=disable_crossover_heuristic,
        disable_heuristics=disable_heuristics,
        extra_params=extra_scip_params,
    )
    # Actual run
    result = SCIP.solve_cbf_with_scip_sdp(
        cbf_path;
        time_limit,
        gap=rel_gap,
        absgap=gap,
        verbose=verbose,
        sdp_mode=scip_sdp_mode,
        presolving=presolve,
        symmetry=symmetry,
        disable_crossover_heuristic=disable_crossover_heuristic,
        disable_heuristics=disable_heuristics,
        extra_params=extra_scip_params,
    )
    status = get(_SCIP_STATUS_TO_MOI, result.status, MOI.OTHER_ERROR)
    solution = result.obj_val
    t = result.solve_time
    dual_bound = result.dual_bound
    rel_gap = result.rel_gap
    @show solution, dual_bound, rel_gap
    ord = result.var_values_ordered
    y = criterion == "ACST" ? reshape(ord[1:n^2], n, n) : ord[1:m]

    isfile(cbf_path) && rm(cbf_path, force=true)
    @show y

    # Diagnostics: n_nodes (both modes), n_cuts_found/n_cuts_applied (OA), n_sdp_iters (B&B; nothing if not exposed)
    diagnostics = (; n_nodes=result.n_nodes, n_cuts_found=result.n_cuts_found, n_cuts_applied=result.n_cuts_applied, n_sdp_iters=result.n_sdp_iters)

    @show diagnostics
    if criterion == "AGC"
        present_edges = connected ? Int(floor(2 * m)) : Int(floor(1/2 * m))
        edges, potential_edges = build_graph_connectivity_data(n, present_edges, m, seed=seed, connected=connected)
        L = graph_laplacian(n, edges)
        A = potential_edges_incidence_matrix(n, potential_edges)
        C = L + ones(n, n)
        ub = fill(1.0, m)
        N = !isfinite(N) ? Int(floor(m/2)) : N
    else
        A, C, N, ub, _ =  build_data(seed, m, n, false, corr, zero_one=zero_one, N=N)
    end
    if criterion == "ACST"
        A, L, _, potential_edges = data_ACST(n, seed, use_base_graph=use_base_graph)
        m = size(A, 1)
        n = size(A, 2)
        L += ones(n, n)
        ub = fill(1.0, m)
        N = augment_budget == -1 ? n-1 : augment_budget
        f_check, _ = build_e_criterion(A, L=L, tightened=tightened)
        # OA / numerics: binaries may be fractionally off; round before combinatorial checks.
        graph = Graphs.complete_graph(n)
        n_kn = Graphs.ne(graph)
        y_tree = zeros(Float64, n_kn)
        for (k, (i, j)) in enumerate(potential_edges)
            y_tree[k] = y[i, j]
        end
        lmo = Boscia.ManagedLMO(
            CO.SpanningTreeLMO(graph),
            fill(0.0, n_kn),
            fill(1.0, n_kn),
            collect(1:n_kn),
            n_kn,
        )
        feasible = Boscia.is_linear_feasible(lmo, y_tree)
        scaled_solution = feasible ? f_check(y_tree) : Inf
        @show sum(y_tree)
    else
        f_check, _ = build_e_criterion(A, L=criterion in ["AGC", "EF"] ? C : nothing, tightened=tightened)
        feasible = isfeasible(seed, m, n, criterion, y, corr, ub=ub, N=N)
        scaled_solution = feasible ? f_check(y) : Inf
    end
    @show feasible, scaled_solution


    if boscia_solution !== nothing
        @show boscia_solution
        @show isfeasible(seed, m, n, criterion, boscia_solution, corr, ub=ub, N=N)
        @show f_check(boscia_solution)
        @show abs(f_check(boscia_solution) - f_check(y))/min(abs(f_check(boscia_solution)), abs(f_check(y)))
    end

    if write
        run_mode = scip_sdp_mode == :oa ? "oa" : "bnb"
        df = DataFrame(
            seed=seed, numberOfExperiments=m, numberOfParameters=n, time=t, N=N,
            solution=solution, dual_bound=dual_bound, rel_gap=rel_gap, tightened=tightened,
            scaled_solution=scaled_solution, termination=status, feasible=feasible,
            n_nodes=diagnostics.n_nodes,
            n_cuts_found=diagnostics.n_cuts_found, n_cuts_applied=diagnostics.n_cuts_applied,
            n_sdp_iters=something(diagnostics.n_sdp_iters, missing),
        )
        connection = criterion == "AGC" ? connected ? "connected" : "disconnected" : ""
        scaled = scaled_input ? "_scaled_" : ""
        tighten = tightened ? "_tightened_" : ""
        file_name = joinpath(@__DIR__, "../csv/SCIPSDP/scip_sdp_$(run_mode)_$(criterion)$(scaled)_optimality_$(corr ? "correlated" : "independent")_$(connection)$(tighten)_$(m)_$(n)_$(N)_$(seed).csv")
        isfile(file_name) ? CSV.write(file_name, df, append=false) : CSV.write(file_name, df, writeheader=true)
    end
    return_diagnostics ? (y, diagnostics) : y
end
