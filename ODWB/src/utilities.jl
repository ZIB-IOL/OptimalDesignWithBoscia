# Utilities
"""
    build_data

seed - for the Random functions.
m    - number of experiments.
fusion - boolean deiciding whether we build the fusion or standard problem.
corr - boolean deciding whether we build the independent or correlated data.   
"""
function build_data(seed, m, n, fusion, corr; scaling_C=false)
    @show scaling_C
    # set up
    Random.seed!(seed)
    if corr 
        B = rand(m,n)
        B = B'*B
        @assert isposdef(B)
        D = MvNormal(randn(n),B)
        
        A = rand(D, m)'
        @assert rank(A) == n 
    else 
        A = rand(m,n)
        @assert rank(A) == n # check that A has the desired rank!
    end 
    C_hat = rand(2n, n)
    C = scaling_C ? 1/2n*transpose(C_hat)*C_hat : transpose(C_hat)*C_hat

    @assert rank(C) == n
    
    if fusion
        N = rand(floor(m/20):floor(m/3))
        ub = rand(1.0:m/10, m)
    else
        N = floor(1.5*n)
        u = floor(N/3)
        ub = rand(1.0:u, m)
    end
        
    return A, C, N, ub, C_hat
end

"""
Build LMO for the problems. Used in Boscia and SCIP. 
"""
function build_lmo(o, m, N, ub)
    MOI.set(o, MOI.Silent(), false)
    MOI.empty!(o)
    x = MOI.add_variables(o, m)
    for i in 1:m
        MOI.add_constraint(o, x[i], MOI.GreaterThan(0.0))
        MOI.add_constraint(o, x[i], MOI.LessThan(ub[i]))
        MOI.add_constraint(o, x[i], MOI.Integer()) 
    end
    MOI.add_constraint(
        o,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(m), x), 0.0),
        MOI.EqualTo(N)
    )
    lmo = FrankWolfe.MathOptLMO(o)

    return lmo, x
end

"""
Build Probability Simplex BLMO for Boscia
"""
function build_blmo(m, N, ub)
    simplex_lmo = Boscia.ProbabilitySimplexSimpleBLMO(N)
    blmo = Boscia.ManagedBoundedLMO(simplex_lmo, fill(0.0, m), ub, collect(1:m), m)
    return blmo
end

"""
Build function for the A-criterion. 
"""
function build_a_criterion(A, fusion; μ=1e-4, C=nothing, build_safe=false, long_run=false)
    m, n = size(A) 
    a = m
    if !fusion && m in [100,120]
        μ = 1e-3
    end
    domain_oracle = build_domain_oracle(A, n)

    if fusion && C === nothing
        @error("For the fusion problem, please provide a matrix C.")
    end

    function f_a(x)
        X = fusion ? C + transpose(A)*diagm(x)*A : transpose(A)*diagm(x)*A + Matrix(μ *I, n, n)
        X = Symmetric(X)
        U = cholesky(X)
        X_inv = U \ I
        return LinearAlgebra.tr(X_inv)/a 
    end

    function grad_a!(storage, x)
        X = fusion ? C + transpose(A)*diagm(x)*A : transpose(A)*diagm(x)*A + Matrix(μ *I, n, n)
        X = Symmetric(X*X)
        F = cholesky(X)
        for i in 1:length(x)
            storage[i] = LinearAlgebra.tr(- (F \ A[i,:]) * transpose(A[i,:]))/a
        end
        return storage 
    end 

    function f_a_safe(x)
        if !domain_oracle(x)
            return Inf
        end
        X = fusion ? C + transpose(A)*diagm(x)*A : transpose(A)*diagm(x)*A + Matrix(μ *I, n, n)
        X = Symmetric(X)
        X_inv = LinearAlgebra.inv(X)
        return LinearAlgebra.tr(X_inv)/a 
    end

    function grad_a_safe!(storage, x)
        if !domain_oracle(x)
            return fill(Inf, length(x))        
        end
        X = fusion ? C + transpose(A)*diagm(x)*A : transpose(A)*diagm(x)*A + Matrix(μ *I, n, n)
        X = Symmetric(X*X)
        F = cholesky(X)
        for i in 1:length(x)
            storage[i] = LinearAlgebra.tr(- (F \ A[i,:]) * transpose(A[i,:]))/a
        end
        return storage
    end

    if build_safe
        return f_a_safe, grad_a_safe!
    end

    return f_a, grad_a!
end

"""
Build function for the D-criterion.
"""
function build_d_criterion(A, fusion; μ =0.0, C=nothing, build_safe=false, long_run=false)
    m, n = size(A)
    a = m
    γ = long_run ? m : 1
    domain_oracle = build_domain_oracle(A, n)

    if fusion && C === nothing
        @error("For the fusion problem, please provide a matrix C.")
    end

    function f_d(x)
        X = fusion ? C + transpose(A)*diagm(x)*A : transpose(A)*diagm(x)*A + Matrix(μ *I, n, n)
        X = Symmetric(X * 1/γ)
        return float(-log(det(X))/a)
    end

    function grad_d!(storage, x)
        X = fusion ? C + transpose(A)*diagm(x)*A : transpose(A)*diagm(x)*A + Matrix(μ *I, n, n)
        X= Symmetric(X * 1/γ)
        F = cholesky(X) 
        for i in 1:length(x)        
            storage[i] = 1/a * LinearAlgebra.tr(-(F \ A[i,:] )*transpose(A[i,:])) * 1/γ
        end
        return storage
    end

    function f_d_safe(x)
        if !domain_oracle(x)
            return Inf
        end
        X = fusion ? C + transpose(A)*diagm(x)*A : transpose(A)*diagm(x)*A + Matrix(μ *I, n, n)
        X = Symmetric(X*1/γ)
        return float(-log(det(X))/a)
    end

    function grad_d_safe!(storage, x)
        if !domain_oracle(x)
            return fill(Inf, length(x))
        end
        X = fusion ? C + transpose(A)*diagm(x)*A : transpose(A)*diagm(x)*A + Matrix(μ *I, n, n)
        X= Symmetric(X*1/γ)
        F = cholesky(X) 
        for i in 1:length(x)        
            storage[i] = 1/a * LinearAlgebra.tr(-(F \ A[i,:] )*transpose(A[i,:])) * 1/γ
        end
        # https://stackoverflow.com/questions/46417005/exclude-elements-of-array-based-on-index-julia
        return storage
    end

    if build_safe
        return f_d_safe, grad_d_safe!
    end

    return f_d, grad_d!
end

function build_general_trace(A, p, fusion; C=nothing, μ=0.0, build_safe=false)
    m, n = size(A) 
    a = 1
    domain_oracle = build_domain_oracle(A, n)

    if fusion && C === nothing
        @error("For the fusion problem, please provide a matrix C.")
    end

    function f_gti(x)
        X = fusion ? C + transpose(A)*diagm(x)*A : transpose(A)*diagm(x)*A + Matrix(μ *I, n, n)
        X= Symmetric(X)
        if p == 0
            return -log(det(X))
        end
        return -1/p * log(LinearAlgebra.tr(Symmetric(X^p))) # 1/n *
    end

    function grad_gti!(storage, x)
        X = fusion ? C + transpose(A)*diagm(x)*A : transpose(A)*diagm(x)*A + Matrix(μ *I, n, n)
        X=Symmetric(X)
        a = p == 0 ? -1 : -1/(LinearAlgebra.tr(Symmetric(X^p)))
        X =Symmetric(X^(p-1))
        for i in 1:m
            storage[i] = a* A[i,:]' * X * A[i,:]
        end
        return storage
    end

    function f_gti_safe(x)
        if !domain_oracle(x)
            return Inf
        end
        X = fusion ? C + transpose(A)*diagm(x)*A : transpose(A)*diagm(x)*A + Matrix(μ *I, n, n)
        X= Symmetric(X)
        if p == 0
            return -log(det(1/n*X))
        end
        return -1/p * log(LinearAlgebra.tr(Symmetric(X^p))) # 1/n *
    end

    function grad_gti_safe!(storage, x)
        if !domain_oracle(x)
            return fill(Inf, length(x))        
        end
        X = fusion ? C + transpose(A)*diagm(x)*A : transpose(A)*diagm(x)*A + Matrix(μ *I, n, n)
        X=Symmetric(X)
        a = p == 0 ? -1 : -1/(LinearAlgebra.tr(Symmetric(X^p)))
        X =Symmetric(X^(p-1))
        for i in 1:m
            storage[i] = a* A[i,:]' * X * A[i,:]
        end
        return storage
    end

    if build_safe
        return f_gti_safe, grad_gti_safe!
    end

    return f_gti, grad_gti!
end

"""
Build general matrix means objective: log(ϕ(X))
"""
function build_matrix_means_objective(A,p;μ=0.0, C_hat=nothing, build_safe=false)
    m,n = size(A)
    domain_oracle = build_domain_oracle(A, n)

    function inf_matrix(x)
        X = C_hat===nothing ? A' * diagm(x) * A + Matrix(μ *I, n, n) : C_hat' * diagm(x[m+1:m+2n]) * C_hat + A' * diagm(x[1:m]) * A
        return X
    end

    function f(x)
        X = inf_matrix(x)
        X=Symmetric(X)
        if p == 0
            return -log(det(1/n*X))
        end
        return -1/p * log(LinearAlgebra.tr(Symmetric(X^p))) # 1/n *
    end

    function grad!(storage, x)
        X = inf_matrix(x)
        X=Symmetric(X)
        a = p == 0 ? -1 : -1/(LinearAlgebra.tr(Symmetric(X^p)))
        X =Symmetric(X^(p-1))
        for i in 1:m
            storage[i] = a* A[i,:]' * X * A[i,:]
        end
        return storage
    end

    function f_safe(x)
        if !domain_oracle(x)
            return Inf
        end
        X = inf_matrix(x)
        X=Symmetric(X)
        if p == 0
            @assert det(1/n*X) > 0 "Determinat is $(det(1/n*X)) and x is $(x)"
            return -log(det(1/n*X))
        end
        return -1/p * log(LinearAlgebra.tr(Symmetric(X^p))) # 1/n *
    end

    function grad_safe!(storage, x)
        if !domain_oracle(x)
            return fill(Inf, length(x))        
        end
        X = inf_matrix(x)
        X=Symmetric(X)
        a = p == 0 ? -1 : -1/(LinearAlgebra.tr(Symmetric(X^p)))
        X =Symmetric(X^(p-1))
        for i in 1:m
            storage[i] = a* A[i,:]' * X * A[i,:]
        end
        return storage
    end

    function linesearch(node, f, grad!, gradient, x, d, theta_max, linesearch_workspace, iter_count, jdx, kdx)
        theta = 0.0
        X = inf_matrix(x)
        X_inv = Symmetric(X^(-1))
        w_jk = A[jdx,:]' * X_inv * A[kdx,:] 
        w_j = A[jdx,:]' * X_inv * A[jdx,:] 
        w_k = A[kdx,:]' * X_inv * A[kdx,:] 
        # D Optimality
        if p == 0
            if w_jk^2 < w_k*w_j
                theta_bar = (w_j-w_k) / (2(w_j*w_k - w_jk^2))
                theta = min(node.upper_bounds[jdx] - x[jdx], x[kdx] - node.lower_bounds[kdx], theta_bar)
            else
                theta = min(node.upper_bounds[jdx] - x[jdx], x[kdx] - node.lower_bounds[kdx])
            end
        # A Optimality 
        elseif p == -1
            X_inv2 = Symmetric(X^(-2))
            z_jk = A[jdx,:]' * X_inv2 * A[kdx,:] 
            z_j = A[jdx,:]' * X_inv2 * A[jdx,:] 
            z_k = A[kdx,:]' * X_inv2 * A[kdx,:] 
            a = z_j - z_k
            b = 2*w_jk*z_jk - w_j*z_k - w_k*z_j
            c = w_j - w_k 
            d = w_j*w_k - w_jk^2
            Delta = a*d + b*c
            
            if !isapprox(Delta, 0.0)
                theta = min(-(b+ sqrt(b^2 -a*Delta)) / Delta, node.upper_bounds[jdx] - x[jdx], x[kdx] - node.lower_bounds[kdx])
            elseif isapprox(Delta, 0.0) && !isapprox(b, 0.0)
                theta = min(- a/(2*b), node.upper_bounds[jdx] - x[jdx], x[kdx] - node.lower_bounds[kdx])
            elseif isapprox(Delta, 0.0) && isapprox(b, 0.0) && a > 0 - 1e-3
                theta = x[kdx] - node.lower_bounds[kdx]
            else
                error("Delta and b are zero, Delta: $(Delta) b: $(b) but a is not positive a: $(a)")
            end
        # other criteria    
        else
            theta = FrankWolfe.perform_line_search(
                    FrankWolfe.Adaptive(),
                    iter_count,
                    f,
                    grad!,
                    gradient,
                    x,
                    -d,
                    theta_max,
                    linesearch_workspace,
                    FrankWolfe.InplaceEmphasis(),
            )
        end
        return theta
    end

    if build_safe
        return f_safe, grad_safe!, linesearch
    end 
    return f, grad!, linesearch
end

"""
Rounding heuristics for the continuous and limit solutions
"""
function heuristics(y, s, ub, mode)
    k = length(findall(x-> x!= 0.0, y))
    z = if mode == "cont"
        round.(y)
    elseif mode == "limit"
        ceil.((s-0.5*k)*y)
    else
        Inf
    end
    if z == Inf
        return Inf
    end
    @show z
    @show sum(z)

    if sum( z .> 1.0) == 0 || sum(z .< ub) == 0
        return Inf
    end

    if sum(z) < s
        while sum(z) < s 
            z = add_to_min(z, ub)
        end
    elseif sum(z) > s
        while sum(z) > s 
            z = remove_from_max(z)
        end
    end
    return z
end

function add_to_min(x, u)
    perm = sortperm(x)
    j = findfirst(x->x != 0, x[perm])
    
    for i in j:length(x)
        if x[perm[i]] < u[perm[i]]
            x[perm[i]] += 1
            break
        else
            continue
        end
    end
    return x
end

function remove_from_max(x)
    perm = sortperm(x, rev = true)
    j = findlast(x->x != 0, x[perm])
    
    for i in 1:j
        if x[perm[i]] > 1
            x[perm[i]] -= 1
            break
        else
            continue
        end
    end
    return x
end

"""
Find n linearly independent rows of A to build the starting point.
"""
function linearly_independent_rows(A, m ,n)
    S = []
    for i in 1:m
        S_i= vcat(S, i)
        if rank(A[S_i,:])==length(S_i)
            S=S_i
        end
        if length(S) == n # we only n linearly independent points
            return S
        end
    end 
    return S # then x= zeros(m) and x[S] = 1
end

"""
Build start point used in FrankWolfe and Boscia in case of A-opt and D-opt.
The functions are self concordant and so not every point in the feasible region
is in the domain of f and grad!.
""" 
function build_start_point2(A, m, n, N, ub)
    # Get n linearly independent rows of A
    S = linearly_independent_rows(A,m,n)
    @assert length(S) == n
    
    x = zeros(m)
    E = []
    V = Vector{Float64}[]

    while !isempty(setdiff(S, findall(x-> !(iszero(x)),x)))
        v = zeros(m)
        while sum(v) < N
            idx = isempty(setdiff(S, findall(x-> !(iszero(x)),v))) ? rand(setdiff(collect(1:m), S)) : rand(setdiff(S, findall(x-> !(iszero(x)),v)))
            if !isapprox(v[idx], 0.0)
                @debug "Index $(idx) already picked"
                continue
            end
            v[idx] = min(ub[idx], N - sum(v))
            push!(E, idx)
        end
        push!(V,v)
        x = sum(V .* 1/length(V)) 
    end
    unique!(V)
    a = length(V)
    x = sum(V .* 1/a)
    active_set= FrankWolfe.ActiveSet(fill(1/a, a), V, x)

    return x, active_set, S
end

"""
Create first incumbent for Boscia and custom BB in a greedy fashion.
"""
function greedy_incumbent(A, m, n, N, ub)
    # Get n linearly independent rows of A
    S = linearly_independent_rows(A,m,n)
    @assert length(S) == n

    # set entries to their upper bound
    x = zeros(m)
    x[S] .= ub[S]

    if isapprox(sum(x), N; atol=1e-4, rtol=1e-2)
        return x
    elseif sum(x) > N
        while sum(x) > N
            remove_from_max(x)
        end
    elseif sum(x) < N
        S1 = S
        while sum(x) < N
            jdx = rand(setdiff(collect(1:m), S1))
            x[jdx] = min(N-sum(x), ub[jdx])
            push!(S1,jdx)
            sort!(S1)
        end
    end
    @assert isapprox(sum(x), N; atol=1e-4, rtol=1e-2)
    @assert sum(ub - x .>= 0) == m 
    return x
end

function greedy_incumbent_fusion(A,m,n,N,ub)
    x = zeros(m)
    for i in 1:m
        x[i] = min(ub[i], N-sum(x))
    end
    return x
end

"""
Check if given point is in the domain of f, i.e. X = transpose(A) * diagm(x) * A 
positive definite.
"""
function build_domain_oracle(A, n)
    return function domain_oracle(x)
        S = findall(x-> !iszero(x),x)
        #@show rank(A[S,:]) == n
        return rank(A[S,:]) == n #&& sum(x .< 0) == 0 
    end
end

"""
Check if a point is linear feasible with respect to the original model
"""
function isfeasible(seed, m, n, criterion, x, corr; N=0)
    if criterion in ["A","D","GTI"]
        A, _, N, ub, _ = build_data(seed, m, n, false, corr)
    elseif criterion in ["AF","DF","GTIF"]
        A, C, N, ub, _ = build_data(seed, m, n, true, corr)
    end

   if sum(x) < N - 1e-4
        return false
   elseif sum(x.>=0-1e-4) != m
        return false
   elseif sum(ub - x.>= 0-1e-4) != m
        return false
   elseif criterion in ["D", "A", "GTI"] && m - sum(iszero.(x)) < n 
        return false
   else
        return true
   end
end
