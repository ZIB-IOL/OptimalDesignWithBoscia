# Utilities
"""
    build_data

seed - for the Random functions.
m    - number of experiments.
fusion - boolean deiciding whether we build the fusion or standard problem.
corr - boolean deciding whether we build the independent or correlated data.   
"""
function build_data(seed, m, n, fusion, corr; scaling_C=false, zero_one=false, N=-Inf)
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
        N = N == -Inf ? rand(floor(m/20):floor(m/3)) : N
        ub = rand(1.0:m/10, m)
    else
        N = N == -Inf ? floor(1.5*n) : N
        u = floor(N/3)
        ub = rand(1.0:u, m)
    end
        
    if zero_one
        return A, C, N, fill(1.0, m), C_hat
    end

    return A, C, N, ub, C_hat
end

function build_integer_data(seed, m, n, fusion, corr; scaling_C=false, M=5, zero_one=false, N=-Inf)
    rng = StableRNG(seed)
    if corr 
        B = rand(rng, m,n)
        B = B'*B
        @assert isposdef(B)
        D = MvNormal(randn(rng, n),B)
        
        A = round.(rand(rng, D, m)')
        @assert rank(A) == n 
    else 
        A = rand(rng, -M:M, m,n)
        @assert rank(A) == n # check that A has the desired rank!
    end 
    C_hat = rand(rng, -M:M, 2n, n)
    C = scaling_C ? 1/2n*transpose(C_hat)*C_hat : transpose(C_hat)*C_hat

    @assert rank(C) == n
    
    if fusion
        N = N == -Inf ? rand(rng, floor(m/20):floor(m/3)) : N
        ub = rand(rng, 1.0:m/10, m)
    else
        N = N == -Inf ? floor(1.5*n) : N
        u = floor(N/3)
        ub = rand(rng, 1.0:u, m)
    end

    if zero_one
        return A, C, N, fill(1.0, m), C_hat
    end
        
    return A, C, N, ub, C_hat
end

"""
    build_graph_connectivity_data(n_nodes, n_edges, n_potential_edges; seed=nothing, connected=true)

Generate data for the maximum algebraic graph connectivity problem: a random graph and
a set of candidate edges that can be added to it.

**Arguments**
- `n_nodes`: number of vertices (nodes).
- `n_edges`: number of edges in the base graph. Must satisfy `n_edges >= n_nodes - 1` if
  `connected=true`, and `n_edges <= n_nodes*(n_nodes-1)/2`.
- `n_potential_edges`: number of potential (candidate) edges to add; these are drawn
  uniformly from edges not present in the base graph.

**Keyword arguments**
- `seed`: optional integer for reproducible randomness (uses `StableRNG`).
- `connected`: if `true`, the base graph is built as connected (spanning tree plus
  random edges); if `false`, edges are chosen uniformly at random among all possible edges.

**Returns**
- `current_edges`: vector of `(i, j)` with `i < j` for the base graph edges.
- `potential_edges`: vector of `(i, j)` with `i < j` for the candidate edges to add.

Edges are undirected and stored once with the smaller node index first. For algebraic
connectivity you typically use the graph Laplacian of the base graph and consider adding
a subset of `potential_edges` to maximize the second smallest eigenvalue.
"""
function build_graph_connectivity_data(
    n_nodes,
    n_edges,
    n_potential_edges;
    seed=nothing,
    connected=true,
)
    rng = seed === nothing ? Random.GLOBAL_RNG : StableRNG(seed)
    max_edges = n_nodes * (n_nodes - 1) ÷ 2
    @assert n_nodes >= 2 "n_nodes must be at least 2"
    @assert n_edges >= 0 && n_edges <= max_edges "n_edges must be in 0 .. $(max_edges)"
    if connected
        @assert n_edges >= n_nodes - 1 "for connected graph, n_edges must be >= n_nodes - 1"
    end
    @assert n_potential_edges >= 0 "n_potential_edges must be non-negative"

    # All possible undirected edges (i, j) with i < j
    all_edges = [(i, j) for i in 1:n_nodes for j in (i+1):n_nodes]

    # Build base graph
    if n_edges == 0
        current_edges = Tuple{Int,Int}[]
    else
        if connected && n_edges >= n_nodes - 1
            # Spanning tree: random permutation of nodes, then connect consecutive nodes
            perm = randperm(rng, n_nodes)
            tree_edges = [(min(perm[i], perm[i+1]), max(perm[i], perm[i+1])) for i in 1:(n_nodes-1)]
            remaining = setdiff(Set(all_edges), Set(tree_edges))
            remaining = collect(remaining)
            n_extra = n_edges - (n_nodes - 1)
            if n_extra > 0
                @assert length(remaining) >= n_extra "not enough remaining edges"
                idx = randperm(rng, length(remaining))[1:n_extra]
                extra_edges = [remaining[i] for i in idx]
                current_edges = vcat(tree_edges, extra_edges)
            else
                current_edges = tree_edges
            end
        else
            # Arbitrary random edges: sample without replacement
            idx = randperm(rng, length(all_edges))[1:n_edges]
            current_edges = [all_edges[i] for i in idx]
        end
    end

    # Potential edges: sample from edges not in current_edges
    current_set = Set(current_edges)
    candidate_pool = [(i, j) for (i, j) in all_edges if !((i, j) in current_set)]
    n_available = length(candidate_pool)
    n_take = min(n_potential_edges, n_available)
    if n_take == 0
        potential_edges = Tuple{Int,Int}[]
    else
        idx = randperm(rng, n_available)[1:n_take]
        potential_edges = [candidate_pool[i] for i in idx]
    end

    return current_edges, potential_edges
end

"""
    graph_laplacian(n_nodes, edges)

Return the Laplacian matrix L (n_nodes × n_nodes) for the graph with the given edges.
`edges` is a vector of `(i, j)` with 1 ≤ i < j ≤ n_nodes. L = ∑_{(i,j)∈edges} (e_i - e_j)(e_i - e_j)'.
"""
function graph_laplacian(n_nodes, edges)
    L = zeros(Float64, n_nodes, n_nodes)
    for (i, j) in edges
        L[i, i] += 1
        L[j, j] += 1
        L[i, j] -= 1
        L[j, i] -= 1
    end
    return L
end

"""
    potential_edges_incidence_matrix(n_nodes, potential_edges)

Return a sparse matrix A of size (length(potential_edges) × n_nodes) where each row
corresponds to a potential edge (i, j): row has a_i = 1, a_j = -1, and 0 elsewhere
(the vector e_i - e_j). So adding a potential edge with weight x contributes
x * (row' * row) to the graph Laplacian.
"""
function potential_edges_incidence_matrix(n_nodes, potential_edges; weights=ones(n_nodes, n_nodes))
    m = length(potential_edges)
    I = Int[]
    J = Int[]
    V = Float64[]
    sizehint!(I, 2 * m)
    sizehint!(J, 2 * m)
    sizehint!(V, 2 * m)
    for (r, (i, j)) in enumerate(potential_edges)
        push!(I, r, r)
        push!(J, i, j)
        push!(V, weights[i, j], -weights[i, j])
    end
    return SparseArrays.sparse(I, J, V, m, n_nodes)
end

"""
Build LMO for the problems. Used in Boscia and SCIP. 
"""
function build_lmo(o, m, N, ub; silent=false)
    MOI.set(o, MOI.Silent(), silent)
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
        MOI.EqualTo(Float64(N))
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

"""
Build the E-criterion and its smoothed version.
"""
function build_e_criterion(A; L=nothing, tightened=false, N=Inf)
    m, n = size(A)
    function inf_matrix(x)
        return L === nothing ? Symmetric(A' * diagm(x) * A) : Symmetric(L + A' * diagm(x) * A)
    end

    function f(x)
        X = inf_matrix(x)   
        #return (-1) * minimum(eigvals(X))  
        return (-1) * LinearAlgebra.eigmin(X)  
    end

    function sub_grad!(storage, x)
        X = inf_matrix(x)
        λ, V = eigen(X)
        λ_min = minimum(λ)
         # Use both relative and absolute tolerance (similar to isapprox)
         tolerance = max(1e-10 * abs(λ_min), 1e-10)
         # Count eigenvalues within tolerance of the minimum
         mult= count(λ_i -> abs(λ_i - λ_min) <= tolerance, λ)
         for i in 1:mult 
            push!(storage, -(A * V[:, i]).^2)
         end
        #storage .= -(A * V[:, 1]).^2
        return storage
    end

    function generate_smoothing_function(μ)

        function f_mu(x)
            X = inf_matrix(x)
            λ = eigvals(X)
            add_on = tightened ? μ/(n - N + 1) * sum(x[i] * norm(A[i, :], 2)^2 for i in 1:m) : 0.0
            #return μ * log(sum(exp.(-λ ./ μ))) - μ * log(n) # logsumexp(X)
            return μ * LogExpFunctions.logsumexp(-λ ./ μ) - μ * log(n) + add_on
        end

        function grad_mu!(storage, x)
            X = inf_matrix(x)
            λ, V = eigen(X)
            frac = - 1/exp(LogExpFunctions.logsumexp(-λ ./ μ))
            add_on = tightened ? μ/(n - N + 1) * norm.(eachrow(A), 2).^2 : 0.0
            storage .= frac * sum(LogExpFunctions.xexpy.((A * V[:,j]).^2 , -λ[j]/ μ)  for j in 1:n) .+ add_on
            return storage
        end
        return f_mu, grad_mu!
    end 

    return f, sub_grad!, generate_smoothing_function
end

function build_general_log_trace(A, p, fusion; C=nothing, μ=0.0, build_safe=false)
    m, n = size(A) 
    a=1
    domain_oracle = build_domain_oracle(A, n)
    @assert p > 0

    if fusion && C === nothing
        @error("For the fusion problem, please provide a matrix C.")
    end

    function f_gti(x)
        X = fusion ? C + transpose(A)*diagm(x)*A : transpose(A)*diagm(x)*A + Matrix(μ *I, n, n)
        X= Symmetric(X)
        U = cholesky(X)
        X_inv = U \ I
        return log(LinearAlgebra.tr(Symmetric(X_inv^(p)))) # 1/n *
    end

    function grad_gti!(storage, x)
        X = fusion ? C + transpose(A)*diagm(x)*A : transpose(A)*diagm(x)*A + Matrix(μ *I, n, n)
        X=Symmetric(X)
        U = cholesky(X)
        X_inv = U \ I
        a = -p/(LinearAlgebra.tr(Symmetric(X_inv^(p))))
        X =Symmetric(X_inv^(p+1))
        for i in 1:m
            storage[i] = a * A[i,:]' * X * A[i,:]
        end
        return storage
    end

    function f_gti_safe(x)
        if !domain_oracle(x)
            return Inf
        end
        X = fusion ? C + transpose(A)*diagm(x)*A : transpose(A)*diagm(x)*A + Matrix(μ *I, n, n)
        X= Symmetric(X)
        U = cholesky(X)
        X_inv = U \ I
        return log(LinearAlgebra.tr(Symmetric(X_inv^(p)))) # 1/n *
    end

    function grad_gti_safe!(storage, x)
        if !domain_oracle(x)
            return fill(Inf, length(x))        
        end
        X = fusion ? C + transpose(A)*diagm(x)*A : transpose(A)*diagm(x)*A + Matrix(μ *I, n, n)
        X=Symmetric(X)
        U = cholesky(X)
        X_inv = U \ I
        a = -p/(LinearAlgebra.tr(Symmetric(X_inv^(p))))
        X =Symmetric(X_inv^(p+1))
        for i in 1:m
            storage[i] = a * A[i,:]' * X * A[i,:]
        end
        return storage
    end

    if build_safe
        return f_gti_safe, grad_gti_safe!
    end

    return f_gti, grad_gti!
end

function build_general_trace(A, p, fusion; C=nothing, μ=0.0, build_safe=false)
    m, n = size(A) 
    a=1
    domain_oracle = build_domain_oracle(A, n)
    @assert p > 0

    if fusion && C === nothing
        @error("For the fusion problem, please provide a matrix C.")
    end

    function f_gti(x)
        X = fusion ? C + transpose(A)*diagm(x)*A : transpose(A)*diagm(x)*A + Matrix(μ *I, n, n)
        X= Symmetric(X)
        U = cholesky(X)
        X_inv = U \ I
        return LinearAlgebra.tr(Symmetric(X_inv^(p))) # 1/n *
    end

    function grad_gti!(storage, x)
        X = fusion ? C + transpose(A)*diagm(x)*A : transpose(A)*diagm(x)*A + Matrix(μ *I, n, n)
        X =Symmetric(X)
        U = cholesky(X)
        X_inv = U \ I
        X =Symmetric(X_inv^(p+1))
        for i in 1:m
            storage[i] = (-p) * A[i,:]' * X * A[i,:]
        end
        return storage
    end

    function f_gti_safe(x)
        if !domain_oracle(x)
            return Inf
        end
        X = fusion ? C + transpose(A)*diagm(x)*A : transpose(A)*diagm(x)*A + Matrix(μ *I, n, n)
        X= Symmetric(X)
        U = cholesky(X)
        X_inv = U \ I
        return LinearAlgebra.tr(Symmetric(X_inv^(p))) # 1/n *
    end

    function grad_gti_safe!(storage, x)
        if !domain_oracle(x)
            return fill(Inf, length(x))        
        end
        X = fusion ? C + transpose(A)*diagm(x)*A : transpose(A)*diagm(x)*A + Matrix(μ *I, n, n)
        X =Symmetric(X)
        U = cholesky(X)
        X_inv = U \ I
        X =Symmetric(X_inv^(p+1))
        for i in 1:m
            storage[i] =(-p) * A[i,:]' * X * A[i,:]
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

function add_to_min2(x,u)
    perm = sortperm(x)
    
    for i in perm
        if x[i] < u[i]
            x[i] += 1
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
function linearly_independent_rows(A, m ,n, ub=nothing)
    S = []
    for i in 1:m
        if ub !== nothing && iszero(ub[i])
            continue
        end
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
Build domain feasible for any node.
"""
function build_domain_point_function(domain_oracle, A, N, int_vars, initial_lb, initial_ub)
    return function domain_point(local_bounds)
        lb = copy(initial_lb)
        ub = copy(initial_ub)
        for idx in int_vars
            if haskey(local_bounds.lower_bounds, idx)
                lb[idx] = max(initial_lb[idx], local_bounds.lower_bounds[idx])
            end
            if haskey(local_bounds.upper_bounds, idx)
                ub[idx] = min(initial_ub[idx], local_bounds.upper_bounds[idx])
            end
        end
        # Node itself infeasible
        if sum(lb) > N 
            println("Node itself infeasible")
            return nothing
        end
        # No intersection between node and domain
        if !domain_oracle(ub)
            println("No intersection node and domain")
            return nothing
        end
        x = lb
        m, n = size(A)
        S = linearly_independent_rows(A, m, n, ub=.!(iszero.(ub)))

        while sum(x) <= N
            if sum(x) == N 
                if domain_oracle(x)
                    return x 
                else 
                    println("No intersection node and domain")
                    return nothing 
                end 
            end
            if !iszero(ub[S]-x[S])
                y = add_to_min2(x[S], ub[S])
                x[S] = y
            else
                x = add_to_min2(x, ub)
            end
        end
    end
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
function isfeasible(seed, m, n, criterion, x, corr; N=0, ub=nothing)
    if criterion in ["A","D","GTI"]
        A, _, N, ub, _ = build_data(seed, m, n, false, corr)
    elseif criterion in ["AF","DF","GTIF"]
        A, C, N, ub, _ = build_data(seed, m, n, true, corr)
    end

   if sum(x) < N - 1e-4
        return false
   elseif sum(x.>=0-1e-4) != m
        return false
   elseif ub !== nothing && sum(ub - x.>= 0-1e-4) != m
        return false
   elseif criterion in ["D", "A", "GTI"] && m - sum(iszero.(x)) < n 
        return false
   else
        return true
   end
end
