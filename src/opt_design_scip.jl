### Optimal design with SCIP
function solve_opt_scip(seed, m, n, time_limit, criterion, corr; write=true, verbose=true)
    if criterion == "A"
       error("SCIP OA does not work with A-opt")
    elseif criterion == "D"
        error("SCIP OA does not work with D-opt")
    elseif criterion == "AF"
        A, C, N, ub =build_data(seed, m, n, true, corr)
        f, grad! = build_a_criterion(A, true, C=C)
    elseif criterion == "DF"
        A, C, N, ub =build_data(seed, m, n, true, corr)
        f, grad! = build_d_criterion(A, true, C=C)
    else
        error("Invalid criterion!")
    end

    # precompilation
    lmo, epigraph_ch, x, lmo_check = build_scip_optimizer(m, N, ub, 10, f, grad!, verbose)
    MOI.optimize!(lmo.o)
        
    lmo, epigraph_ch, x, lmo_check = build_scip_optimizer(m, N, ub, time_limit, f, grad!, verbose)

    MOI.optimize!(lmo.o)
    time_scip = MOI.get(lmo.o, MOI.SolveTimeSec())
    vars_scip = MOI.get(lmo.o, MOI.VariablePrimal(), x)
    @assert Boscia.is_linear_feasible(lmo_check.o, vars_scip)
    solution_scip = f(vars_scip)
    termination_scip = String(string(MOI.get(lmo.o, MOI.TerminationStatus())))
    ncalls_scip = epigraph_ch.ncalls
    dual_objective = MOI.get(lmo.o, MOI.ObjectiveBound())
    @show dual_objective

    type = corr ? "correlated" : "independent"

    if write 
        df = DataFrame(seed=seed, numberOfExperiments=m, numberOfParameters=n, N=N,time=time_scip, solution=solution_scip, dual=dual_objective, termination=termination_scip, calls=ncalls_scip)
        file_name = "/scratch/opt/dhendryc/MasterThesis/optDesign/csv/SCIP/scip_" * criterion * "_" * string(m) * "_" * type * "_optimality" * ".csv"
        if !isfile(file_name)
            CSV.write(file_name, df, append=true, writeheader=true)
        else 
            CSV.write(file_name, df, append=true)
        end
    end
    return vars_scip
end

function build_scip_optimizer(m, N, ub, limit, f, grad!, verbose)
    o = SCIP.Optimizer()
    MOI.set(o, MOI.TimeLimitSec(), limit)
    MOI.set(o, MOI.Silent(), !verbose)
    lmo, x = build_lmo(o, m, N, ub)
    z_i = MOI.add_variable(lmo.o)
    epigraph_ch = GradientCutHandler(lmo.o, f, grad!, zeros(length(x)), z_i, x, 0)
    SCIP.include_conshdlr(lmo.o, epigraph_ch; needs_constraints=false, name="handler_gradient_cuts")
    MOI.set(lmo.o, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), 1.0 * z_i)    
    
    # lmo to verify feasibility of solution after optimization
    o_check = SCIP.Optimizer()
    lmo_check, _ = build_lmo(o_check, m, N, ub)
    z_i = MOI.add_variable(lmo_check.o)
    MOI.set(lmo_check.o, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), 1.0 * z_i)    
    
    return lmo, epigraph_ch, x, lmo_check
end



  

