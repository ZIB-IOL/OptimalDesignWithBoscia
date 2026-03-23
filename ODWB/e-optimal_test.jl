using ODWB
using Boscia
using FrankWolfe

criterion = "E"
seed = 5 # 4
connected = true
if criterion == "AGC"
    n = 80
    m = 2 * n
    N = Int(floor(m/2))
    edges, potential_edges = ODWB.build_graph_connectivity_data(n, 2 * m, m, seed=seed, connected=connected)
    L = ODWB.graph_laplacian(n, edges) + ones(n, n)
    A = ODWB.potential_edges_incidence_matrix(n, potential_edges)
else
    m = 50
    n = Int(floor(sqrt(m)))
    N = Int(floor(1.5 * n * log(n)))
    #N = Int(floor(3n/4))
end
corr = false

#ENV["JULIA_DEBUG"] = "Boscia"

zero_one = true
scale = Inf

start = isfinite(scale) ? 1e-2/(2 *log(n)) : m/50
min = isfinite(scale) ? 1e-6/(2 * log(n)) : exp10(-10/m) 

#start =m/50 # m/10
#min = exp10(-200/m) #exp10(-100/m)

scale = Inf

@show start, min

x, result = ODWB.solve_opt(
    seed, 
    m, 
    n, 
    120, 
    criterion, 
    corr, 
    full_callback=false, 
    write=false, 
    use_BPCG=false,
    verbose=true, 
    use_scip=false,
    zero_one=zero_one, 
    fw_verbose=false,
    ls_secant=false, 
    N=N, 
    use_tightening=false,
    use_sub_grad_info=true,
    smoothing_start=start,
    smoothing_min=min,
    tightened=false,
    scale=scale,
    use_exclusion_criterion=true,
    use_dual_exclusion_criterion=true,
    use_dual_tightening=false,
    start_epsilon=1e-4,
    min_epsilon=1e-7,
)

@show x
@show findall(x-> x == 1, x) 

#= x_e, result_e = ODWB.solve_opt(
    seed, 
    m, 
    n, 
    300, 
    criterion, 
    corr, 
    full_callback=false, 
    write=false, 
    verbose=true, 
    zero_one=zero_one, 
    use_BPCG=false,
    fw_verbose=false, 
    ls_secant=false, 
    smoothing_start=5.0, 
    smoothing_min=1e-1, 
    N=N, 
    use_heuristics=true,
    use_exclusion_criterion=true,
    use_sub_grad_info=true,
)  =#

println("SCIP SDP")

y_bnb = ODWB.solve_opt_scip_sdp(
    seed, 
    m, 
    n, 
    120, 
    criterion, 
    corr, 
    write=false, 
    verbose=true, 
    scip_sdp_mode=:bnb,
    zero_one=zero_one, 
    N=N,
    tightened=false,
    scale=scale,
    rel_gap = 0.0,
    )

    @show findall(y-> y == 1, y_bnb)

#=y_oa = ODWB.solve_opt_scip_sdp(
    seed, 
    m, 
    n, 
    120, 
    criterion, 
    corr, 
    write=false, 
    verbose=true, 
    scip_sdp_mode=:oa,
    zero_one=zero_one, 
    N=N,
    tightened=true,
    scale=scale,
    )

    @show findall(y-> y == 1, y_oa)
    =#
#@show findall(y-> y == 0, y)

#if !any(isnan.(y))
#    @show f(x), f(y), abs(f(x) - f(y))/min(abs(f(x)), abs(f(y)))
#    @show "SCIP SDP solution is better: " f(y) <= f(x)
#end

#=
A, _, N, ub, _ = ODWB.build_integer_data(seed, m, n, false, corr)
lmo = ODWB.build_blmo(m, N, ub)

f, generate_smoothing_function = ODWB.build_e_criterion(A)

x, _, result = Boscia.solve(f, nothing, lmo; 
mode = Boscia.SMOOTHING_MODE,
settings_bnb = Boscia.settings_bnb(verbose=true, print_iter=100, time_limit=60),
settings_smoothing = Boscia.settings_smoothing(mode=Boscia.SMOOTHING_MODE, generate_smoothing_objective = generate_smoothing_function),## , smoothing_start=5.0, smoothing_min=1.0
settings_frank_wolfe = Boscia.settings_frank_wolfe(mode=Boscia.SMOOTHING_MODE, max_fw_iter=1000, fw_verbose=false, line_search=FrankWolfe.Adaptive()),
)=#
