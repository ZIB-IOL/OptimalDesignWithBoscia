using ODWB
using Boscia
using FrankWolfe

seed = 5 # 4
m = 50
n = Int(floor(sqrt(m)))
corr = false
N = Int(floor(1.5 * n * log(n)))

integer_data = false
zero_one = true

A, C, N, ub, _ = integer_data ? ODWB.build_integer_data(seed, m, n, false, corr, zero_one=zero_one, N=N) : ODWB.build_data(seed, m, n, false, corr, zero_one=zero_one, N=N)
f, sub_grad!, _ = ODWB.build_e_criterion(A)
@show A, N


x, result = ODWB.solve_opt(seed, m, n, 60, "E", corr, full_callback=false, write=false, verbose=true, integer_data=integer_data, zero_one=zero_one, fw_verbose=false, ls_secant=true, smoothing_start=5.0, smoothing_min=1e-1, N=N, use_heuristics=true)

println("Pajarito")
y = ODWB.solve_opt_pajarito(seed, m, n, 300, "E", corr, write=false, verbose=true, integer_data=integer_data, boscia_solution=x, zero_one=zero_one, N=N)

if !any(isnan.(y))
    @show f(x), f(y), abs(f(x) - f(y))/min(abs(f(x)), abs(f(y)))
end

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
