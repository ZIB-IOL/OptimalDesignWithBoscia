using ODWB
using LinearAlgebra
using Boscia

for m in 50:20:200
    n = Int(floor(m/10))
    @show m, n
    N = Int(floor(1.5 * n))
    A, _, N, _ = ODWB.build_data(1, m, n, false, false)
    lmo = ODWB.build_blmo(m, N, fill(1.0, m))
    for _ in 1:10
        v = rand(m)
        x = Boscia.compute_extreme_point(lmo, v)
        X = A' * diagm(x) * A
        eigs = reverse(eigvals(-X))
        max_sing = svdvals(-A' * A)[1]
        mu_start = m/20
        mu_decay = 0.7
        mu_min = exp10(-100/m)
        for depth in 1:min(m, 100)
            μ = max(mu_start * (mu_decay)^(depth - 1), mu_min)
            epsilon = max(1e-2 * 0.8^(depth - 1), 1e-6)
            delta = epsilon/10
            @show μ, epsilon, delta
            for k in 1:n
                lhs = (sqrt(2) * (n - k) * exp((eigs[k] - eigs[1])/μ))/(sum(exp((eigs[j] - eigs[1])/μ) for j in 1:k))
                rhs = delta/max_sing
                if !isfinite(lhs)
                    @show k, eigs[k], eigs[1], μ
                end
                #@show k, lhs, rhs
                if lhs <= rhs
                    @show k, lhs, rhs
                    break
                end
            end 
        end
end
    println("\n")
end