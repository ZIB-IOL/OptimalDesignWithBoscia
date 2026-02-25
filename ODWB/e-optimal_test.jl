using ODWB
using Boscia
using FrankWolfe

criterion = "AGC"
seed = 5 # 4
connected = true
if criterion == "AGC"
    n = 50
    m = 2 * n
    N = Int(floor(m/2))
else
    m = 50
    n = Int(floor(sqrt(m)))
    N = Int(floor(1.5 * n * log(n)))
end
corr = false

#ENV["JULIA_DEBUG"] = "Boscia"

zero_one = true

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
    use_tightening=true,
    use_sub_grad_info=true,
    smoothing_start=m/25,
    smoothing_min=1e-1,
)

@show x
@show findall(x-> x == 0, x) 

x_e, result_e = ODWB.solve_opt(
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
) 

println("SCIP SDP")

y_bnb = ODWB.solve_opt_scip_sdp(
    seed, 
    m, 
    n, 
    300, 
    criterion, 
    corr, 
    write=false, 
    verbose=true, 
    scip_sdp_mode=:bnb,
    zero_one=zero_one, 
    N=N,
    )

y_oa = ODWB.solve_opt_scip_sdp(
    seed, 
    m, 
    n, 
    300, 
    criterion, 
    corr, 
    write=false, 
    verbose=true, 
    scip_sdp_mode=:oa,
    zero_one=zero_one, 
    N=N,
    )
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
