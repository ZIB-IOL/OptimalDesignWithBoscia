"""
    set_objective!(model, sense, objective_expr; [solve=false])

Replace the objective of an existing JuMP model without rebuilding variables or constraints.
Useful when solving many problems that share the same constraint set and differ only in the objective.

- `model`: JuMP model (with variables and constraints already added).
- `sense`: `MOI.MIN_SENSE` or `MOI.MAX_SENSE`.
- `objective_expr`: JuMP expression (e.g. a variable ref, or `sum(c[i]*x[i] for i in 1:n)`).
- `solve`: if `true`, call `optimize!(model)` after setting the objective.

Returns the model. If `solve=true`, you can then use `termination_status(model)`, `objective_value(model)`, `value.(variables)`, etc.

# Example (SDP with fixed constraints, varying objective)

```julia
using JuMP
using Hypatia
const MOI = MathOptInterface

# Build model once: variables and constraints
opt = optimizer_with_attributes(Hypatia.Optimizer, MOI.Silent() => true)
model = Model(opt)
@variable(model, λ)
@variable(model, Z[1:n, 1:n], PSD)
@constraint(model, sum(Z[i,i] for i in 1:n) == 1)
# ... add all λ ≥ v_i' Z v_i constraints ...

# Solve with different objectives without rebuilding
for (name, obj_expr) in [("min λ", λ), ("min 2λ", 2λ)]
    set_objective!(model, MOI.MIN_SENSE, obj_expr; solve=true)
    @show name, objective_value(model)
end
```
    set_objective_and_solve!(model, sense, objective_expr)

Convenience: same as `set_objective!(model, sense, objective_expr; solve=true)`.
"""
function set_objective_and_solve!(
    model::JuMP.Model,
    sense::MOI.OptimizationSense,
    objective_expr,
)
    JuMP.set_objective_sense(model, sense)
    JuMP.set_objective_function(model, objective_expr)
    JuMP.optimize!(model)
    return model
end

function model_exclusion(A, m, n, UB, LB, M; u=fill(1.0, m), x=fill(0.0, m), vdix=-1)
    fixed_indices = Int64[]
    opt = optimizer_with_attributes(Mosek.Optimizer, MOI.Silent() => true)
    model = Model(opt)
    @variable(model, Z[1:n, 1:n])
    @constraint(model, sum(Z[i,i] for i in 1:n) == 1)
    @constraint(model, Z in PSDCone())
    for i in 1:m
        @constraint(model, M * A[i, :]' * Z * A[i, :] <= UB)
    end
    @show UB
   
    for i in 1:m
        if isapprox(x[i], 0.0, atol=1e-6) || u[i] == 0.0 || i == vdix
            continue
        end
        objective = M * A[i, :]' * Z * A[i, :] 
        set_objective_and_solve!(model, MOI.MAX_SENSE, objective)
        obj = objective_value(model)
        #@show i, obj, LB
        if obj <= LB
            @show i, obj, LB
            push!(fixed_indices, i)
        end
    end
   # @show fixed_indices
    return model, fixed_indices
end

function build_exclusion_branch_callback(A, N, f, sub_grad!)
    return function branch_callback(tree,
        node, vdix)
        m, n = size(A)
        if node.depth > n
            return false, false
        end
        u = fill(1.0, m)
        for i in tree.root.problem.integer_variables
            local_ub = get(node.local_bounds.upper_bounds, i, Inf)
            local_lb = get(node.local_bounds.lower_bounds, i, -Inf)
            if local_ub == 0.0 || local_lb == 1.0
                u[i] = 0.0
            end
        end
        y = copy(node.active_set.x/N) 
        @show vdix
        @show y
        Y = A' * diagm(y) * A
        λ, V = eigen(Y)
        if !isreal(λ[1])
            return false, false
        end
        λ_min = minimum(λ)
        tolerance = max(1e-10 * abs(λ_min), 1e-10)
        mult = count(λ_i -> abs(λ_i - λ_min) <= tolerance, λ)
        fx = -f(y)
        W = Symmetric(sum(V[:, j] * V[:, j]' for j in 1:mult)) #+ I(n)
        W -= n/2 * minimum(eigvals(W)) * I(n)
        W = 1/LinearAlgebra.tr(W) * W 

        @show mult

        UB = maximum(A[j,:]' * V[:,k] * V[:,k]' * A[j,:] for k in 1:mult for j in 1:m)

        if UB < fx
            return false, false
        end

        #UB = maximum(A[j,:]' * W * A[j,:] for j in 1:m) # * N
        _, fixed_indices = model_exclusion(A, m, n, UB, fx, 1.0, u=u, x=y, vdix=vdix)

        @show fixed_indices

        for i in fixed_indices
            push!(node.local_bounds.upper_bounds, (i => 0.0))
            #node.local_bounds.upper_bounds[i] = 0.0
        end
        return false, false
    end
end

function build_tightened_branch_callback(A, N, f, sub_grad!; L=nothing)
    return function branch_callback(tree,
        node, vdix)
        m, n = size(A)
        if node.depth > m
            return false, false
        end

        # collecting current fixings
        l = zeros(m)
        u = ones(m)
        fixed = []
        N_star = Int(N)
        for i in tree.root.problem.integer_variables
            local_ub = get(node.local_bounds.upper_bounds, i, Inf)
            local_lb = get(node.local_bounds.lower_bounds, i, -Inf)
            l[i] = isfinite(local_lb) ? local_lb : 0.0
            u[i] = isfinite(local_ub) ? local_ub : 1.0
            if isfinite(local_lb) || isfinite(local_ub)
                push!(fixed, i)
            end
            N_star = isfinite(local_lb) ? N_star - 1 : N_star
        end

        # generate dual z
        y = node.active_set.x
        Y = L === nothing ? A' * diagm(y) * A : L + A' * diagm(y) * A
        λ, V = eigen(Y)
        if !isreal(λ[1])
            return false, false
        end
        λ_min = minimum(λ)
        @show λ_min
        @show tree.root.options[:original_objective](y)
        tolerance = max(1e-10 * abs(λ_min), 1e-10)
        mult = count(λ_i -> abs(λ_i - λ_min) <= tolerance, λ)
        @show mult
        Z = Symmetric(sum(V[:, j] * V[:, j]' for j in 1:mult)) #+ I(n)
        Z = Z/tr(Z)
        v = [A[i,:]' * Z * A[i,:] for i in setdiff(1:m, fixed)]
        v = sort(v)
        t = v[Int(N_star)] 
      #= k = 0
        t = Inf
        for j in 1:mult 
            Z_temp = V[:, j] * V[:, j]'
            Z_temp = Z_temp/tr(Z_temp)
            v_temp = [A[i,:]' * Z_temp * A[i,:] for i in setdiff(collect(1:m), fixed)]
            v_temp = sort(v_temp)
            t_temp = v_temp[Int(N_star)]
            #@show t1
            #@show v1
            if t_temp < t
                t = t_temp
                k = j
            end
        end

       Z = V[:, k] * V[:, k]'
        Z = Z/tr(Z) =#
        # pick dual t (only for free variables)
        v = [A[i,:]' * Z * A[i,:] for i in 1:m]

        # compute α and β
        α = max.(0, t - v[i] for i in 1:m)
        β = max.(0, v[i] - t for i in 1:m)
        # perform the dual tightening
        UB = L === nothing ? t * N - α' * l + β' * u : t * N - α' * l + β' * u + dot(Z, L)
        @assert UB > λ_min " UB: $(UB) λ_min: $(λ_min)"
        if UB < -tree.incumbent
            return false, false
        end

        prune_left = false
        prune_right = false

        @show tree.incumbent
        @show UB
        fixed_to_one = []
        fixed_to_zero = []

        for i in tree.root.problem.integer_variables
            if i in fixed || i == vdix
                continue
            end
            right_bound = UB - α[i]
            left_bound = UB - β[i]
            @show left_bound, right_bound
            if right_bound <= -tree.incumbent 
                push!(node.local_bounds.upper_bounds, (i => 0.0))
                prune_right = i == vdix ? true : false
                push!(fixed_to_zero, i)
            elseif left_bound <= -tree.incumbent 
                push!(node.local_bounds.lower_bounds, (i => 1.0))
                prune_left = i == vdix ? true : false
                push!(fixed_to_one, i)
            end
        end
        @show fixed_to_one
        @show fixed_to_zero

        @assert length(fixed_to_one) < N
        @assert length(fixed_to_zero) < m - N
        return prune_left, prune_right
    end
end

"""
Compute t(Z) = N_star-th smallest of v_i = a_i' Z a_i over *free* indices.
`fixed_mask[i] == true` means i is fixed (so i is free iff `!fixed_mask[i]`).
"""
function t_from_Z(A, Z, fixed_mask::AbstractVector{Bool}, N_star::Int; work=nothing)
    m, _ = size(A)
    nfree = m - count(fixed_mask)
    if work === nothing
        v_free = Vector{Float64}(undef, nfree)
        tmp = zeros(eltype(Z), size(Z, 1))
    else
        v_free = work.v_free
        tmp = work.tmp
    end
    k = 0
    Zsym = Symmetric(Z)
    @inbounds for i in 1:m
        if !fixed_mask[i]
            k += 1
            ai = view(A, i, :)
            mul!(tmp, Zsym, ai)
            v_free[k] = dot(ai, tmp)
        end
    end
    v_view = view(v_free, 1:k)
    partialsort!(v_view, N_star)
    return v_view[N_star]
end

"""
Search Z within the min-eigenspace to make t small.
- tries eigenvector projectors
- tries random rank-1 projectors in the eigenspace
"""
function pick_Z_minimizing_t(A, U, fixed_mask::AbstractVector{Bool}, N_star::Int;
                            n_random::Int=50, rng=Random.default_rng())
    n, mult = size(U)
    Zbest = zeros(eltype(A), n, n)
    tbest = Inf

    # workspace
    v_free_work = Vector{Float64}(undef, length(fixed_mask) - count(fixed_mask))
    tmp_work = zeros(eltype(A), n)
    Z = zeros(eltype(A), n, n)
    work = (v_free=v_free_work, tmp=tmp_work)

    # 1) try each eigenvector projector
    for j in 1:mult
        fill!(Z, 0)
        uj = view(U, :, j)
        LinearAlgebra.BLAS.ger!(1.0, uj, uj, Z)  # Z = uj*uj'
        # trace is 1 already if uj normalized, but safe:
        Z ./= LinearAlgebra.tr(Z)
        t = t_from_Z(A, Z, fixed_mask, N_star; work=work)
        if t < tbest
            tbest = t
            copyto!(Zbest, Z)
        end
    end

    # 2) random rank-1 projectors in eigenspace: Z = U*q*q'*U'
    q = zeros(Float64, mult)
    tmp = zeros(Float64, n)
    for _ in 1:n_random
        randn!(rng, q); q ./= norm(q)
        # tmp = U*q
        mul!(tmp, U, q)
        fill!(Z, 0)
        LinearAlgebra.BLAS.ger!(1.0, tmp, tmp, Z)
        Z ./= LinearAlgebra.tr(Z)
        t = t_from_Z(A, Z, fixed_mask, N_star; work=work)
        if t < tbest
            tbest = t
            copyto!(Zbest, Z)
        end
    end

    return Zbest, tbest
end

function tightening_from_dual(tree, node, vdix, A, L, fixed_mask, N, N_star, l, u,n_random; tighted_to_one=Dict{Int, Int}(), tighted_to_zero=Dict{Int, Int}())
    m, n = size(A)
    T = eltype(A) 
    v_all = zeros(T, m)
    v_free = zeros(T, m) # used as workspace; only first free_count entries are valid

    α = zeros(T, m)
    β = zeros(T, m)

    Zmat = zeros(T, n, n) # workspace for Z (symmetric)
    tmp_qf = zeros(T, n)  # workspace for quadratic forms a_i' Z a_i (works for sparse rows too)

    # Preallocated buffers for printing which variables were newly fixed in a callback call.
    # We store indices in a dense vector and track a count to avoid repeated allocations.
    fixed_to_zero = Vector{Int}(undef, m)
    fixed_to_one = Vector{Int}(undef, m) 
    # generate dual Z
    y = node.active_set.x
    Y = L === nothing ? (A' * Diagonal(y) * A) : (L + A' * Diagonal(y) * A)
    λ, V = eigen(Y)
    if !isreal(λ[1])
        return 
    end
    λ_min = minimum(λ)
    tolerance = max(1e-10 * abs(λ_min), 1e-10)
    mult = count(λ_i -> abs(λ_i - λ_min) <= tolerance, λ)

    if n_random > 0
        Zsym, _ = pick_Z_minimizing_t(A, V[:, 1:mult], fixed_mask, N_star; n_random=n_random)
        Zsym = Symmetric(Zsym)
    else
         #Z = sum_{j=1}^{mult} V[:,j] * V[:,j]' , normalized to tr(Z)=1
        fill!(Zmat, zero(T))
        @inbounds for j in 1:mult
            vj = view(V, :, j)
            LinearAlgebra.BLAS.ger!(one(T), vj, vj, Zmat)
        end
        trZ = zero(T)
        @inbounds for i in 1:n
            trZ += Zmat[i, i]
        end
        Zmat ./= trZ
        Zsym = Symmetric(Zmat)
    end

    # compute v_i = a_i' Z a_i for all i, and collect free ones for selecting t
    free_count = 0
    @inbounds for i in 1:m
        ai = view(A, i, :)
        mul!(tmp_qf, Zsym, ai)
        vi = LinearAlgebra.dot(ai, tmp_qf)
        v_all[i] = vi
        if !fixed_mask[i]
            free_count += 1
            v_free[free_count] = vi
        end
    end

    # pick dual t (only for free variables): t = N_star-th smallest among free vars
    v_free_view = view(v_free, 1:free_count)
    partialsort!(v_free_view, N_star)
    t = v_free_view[N_star]

    # compute α and β without broadcasting allocations
    @inbounds for i in 1:m
        vi = v_all[i]
        if t > vi
            α[i] = t - vi
            β[i] = zero(T)
        elseif vi > t
            α[i] = zero(T)
            β[i] = vi - t
        else
            α[i] = zero(T)
            β[i] = zero(T)
        end
    end

    UB = if L === nothing
        t * N - LinearAlgebra.dot(α, l) + LinearAlgebra.dot(β, u)
    else
        t * N - LinearAlgebra.dot(α, l) + LinearAlgebra.dot(β, u) + LinearAlgebra.dot(Zsym, L)
    end
    @assert UB > λ_min " UB: $(UB) λ_min: $(λ_min)"
    if UB < -tree.incumbent
        return 
    end

    zc = 0
    oc = 0
    for i in 1:m
        if fixed_mask[i] || i == vdix
            continue
        end
        right_bound = UB - α[i]
        left_bound = UB - β[i]
        if right_bound <= -tree.incumbent
            push!(node.local_bounds.upper_bounds, (i => 0.0))
            zc += 1
            fixed_to_zero[zc] = i
        elseif left_bound <= -tree.incumbent
            push!(node.local_bounds.lower_bounds, (i => 1.0))
            oc += 1
            fixed_to_one[oc] = i
        end
    end

    # build up node LMO
    Boscia.build_LMO(
        tree.root.problem.tlmo,
        tree.root.problem.integer_variable_bounds,
        node.local_bounds,
        tree.root.problem.integer_variables,
    )
    if !Boscia.is_linear_feasible(tree.root.problem.tlmo, y) && tree.root.options[:variant] == Boscia.BlendedPairwiseConditionalGradient()
        push!(node.local_bounds.upper_bounds, (vdix => 0.0))
        # build up node LMO
        Boscia.build_LMO(
            tree.root.problem.tlmo,
            tree.root.problem.integer_variable_bounds,
            node.local_bounds,
            tree.root.problem.integer_variables,
        )
        v_left = Boscia.compute_extreme_point(tree.root.problem.tlmo, y)
        delete!(node.local_bounds.upper_bounds, vdix)

        push!(node.local_bounds.lower_bounds, (vdix => 1.0))
        # build up node LMO
        Boscia.build_LMO(
            tree.root.problem.tlmo,
            tree.root.problem.integer_variable_bounds,
            node.local_bounds,
            tree.root.problem.integer_variables,
        )
        v_right = Boscia.compute_extreme_point(tree.root.problem.tlmo, y)
        delete!(node.local_bounds.lower_bounds, vdix)

        Boscia.build_LMO(
            tree.root.problem.tlmo,
            tree.root.problem.integer_variable_bounds,
            node.local_bounds,
            tree.root.problem.integer_variables,
        )

        active_set = FrankWolfe.ActiveSet([(0.5, v_left), (0.5, v_right)])
        node.active_set = active_set
    end

    if zc > 0 || oc > 0
        fixed_to_zero_view = view(fixed_to_zero, 1:zc)
        fixed_to_one_view = view(fixed_to_one, 1:oc)
        @show fixed_to_zero_view
        @show fixed_to_one_view
        zc > 0 ? push!(tighted_to_zero, node.id => zc) : nothing
        oc > 0 ? push!(tighted_to_one, node.id => oc) : nothing
    end
end

"""
    build_tightened_branch_callback_mem(A, N, f, sub_grad!; L=nothing)

Memory-optimized variant of `build_tightened_branch_callback` that yields the same
branching tightenings (same `t`, same `Z`, same `α`, `β`, same UB comparisons),
but reduces allocations by:
- reusing buffers across callback invocations,
- using `Diagonal(y)` instead of `diagm(y)`,
- avoiding `setdiff`/`fixed` arrays via a boolean mask,
- using `partialsort!` to select the `N_star`-th smallest value among free vars,
- building `Z = sum_{j=1}^{mult} v_j v_j^T` via rank-1 updates into a single matrix.
"""
function build_branch_callback_mem(
    A,
    N,
    f,
    sub_grad!;
    L=nothing,
    print_fixings::Bool=false,
    n_random::Int=10,
    tightening=false,
    tighted_to_one=Dict{Int, Int}(),
    tighted_to_zero=Dict{Int, Int}(),
    processed_tightening_nodes=0,
    number_pruned_nodes=Dict{Int, Int}(),
    processed_pruning_nodes=0,
    rank_based_pruning=false,
    eigenvalue_based_pruning=false,
)
    m, n = size(A)
    T = eltype(A)

    l = zeros(T, m)
    u = ones(T, m)
    fixed_mask = falses(m)

    # Buffers reused across callback invocations (avoid per-node allocations).
    free_indices = Vector{Int}(undef, m)
    V_i = zeros(T, n, n)      # A[vdix,:] * A[vdix,:]'
    G_free = zeros(T, n, n)   # A_free' * A_free
    tmp = zeros(T, n, n)      # workspace for eigmin calls

    return function branch_callback(tree, node, vdix)
        if node.depth > m
            return false, false
        end

        fill!(l, zero(T))
        fill!(u, one(T))
        fill!(fixed_mask, false)
        N_star = Int(N)
        M_0 = L === nothing ? zeros(T, n, n) : copy(L)

        # collecting current fixings
        int_vars = tree.root.problem.integer_variables
        for i in int_vars
            local_ub = get(node.local_bounds.upper_bounds, i, Inf)
            local_lb = get(node.local_bounds.lower_bounds, i, -Inf)
            l[i] = isfinite(local_lb) ? local_lb : zero(T)
            if isfinite(local_lb)
                M_0 += A[i, :] * l[i] * A[i, :]'
            end
            u[i] = isfinite(local_ub) ? local_ub : one(T)
            if isfinite(local_lb) || isfinite(local_ub)
                fixed_mask[i] = true
            end
            N_star = isfinite(local_lb) ? (N_star - 1) : N_star
        end

        prune_left = false
        prune_right = false
        if node.depth > Int(N) && rank_based_pruning
            # Collect free indices without allocating a fresh vector.
            free_count = 0
            @inbounds for i in 1:m
                if !fixed_mask[i]
                    free_count += 1
                    free_indices[free_count] = i
                end
            end
            free_view = view(free_indices, 1:free_count)

            # rank(A_free) == rank(A[free,:]) but the latter avoids forming A_free*A_free'.
            # rank(V_i) is 1 unless A[vdix,:] is (numerically) zero.
            @views a = A[vdix, :]
            r_V = iszero(norm(a)) ? 0 : 1
            r_M0 = rank(M_0)
            # `rank` on sparse SubArray can dispatch through `svdvals!` and fail.
            # Materialize only this small free-row block when A is sparse.
            r_Afree = if free_count == 0
                0
            elseif issparse(A)
                rank(Matrix(@view(A[free_view, :])))
            else
                rank(@view(A[free_view, :]))
            end

            local_rank_l = r_M0 + min(N_star, r_Afree - r_V)
            local_rank_u = r_M0 + min(N_star - 1, r_Afree)
            prune_left = local_rank_l < n
            prune_right = local_rank_u < n
            if prune_left || prune_right
                @show n, r_M0, r_Afree, N_star, local_rank_l, local_rank_u, prune_left, prune_right
                push!(number_pruned_nodes, node.id => prune_left + prune_right)
                processed_pruning_nodes += 1
            end
        elseif eigenvalue_based_pruning && node.depth > Int(N)
            # Collect free indices without allocating a fresh vector.
            free_count = 0
            @inbounds for i in 1:m
                if !fixed_mask[i]
                    free_count += 1
                    free_indices[free_count] = i
                end
            end
            free_view = view(free_indices, 1:free_count)

            # Build V_i and G_free in parameter space (n×n) using preallocated buffers.
            # V_i = a*a',  G_free = A_free' * A_free.
            @views a = A[vdix, :]
            mul!(V_i, a, a', one(T), zero(T))
            if free_count == 0
                fill!(G_free, zero(T))
            else
                A_free_rows = @view(A[free_view, :])
                mul!(G_free, A_free_rows', A_free_rows, one(T), zero(T))
            end

            # Reuse a workspace matrix for eigmin to avoid allocating M_0 + ...
            tmp .= M_0
            tmp .+= G_free
            tmp .-= V_i
            left_eig = eigmin(Symmetric(tmp))

            tmp .= M_0
            tmp .+= G_free
            right_eig = eigmin(Symmetric(tmp))
            prune_left = left_eig < -tree.incumbent
            prune_right = right_eig < -tree.incumbent
            if prune_left || prune_right
                @show n, rank(M_0), free_count, N_star, left_eig, right_eig, prune_left, prune_right
                push!(number_pruned_nodes, node.id => prune_left + prune_right)
                processed_pruning_nodes += 1
            end
        end

        if node.depth < m && tightening
            processed_tightening_nodes += 1
            tightening_from_dual(tree, node, vdix, A, L, fixed_mask, N, N_star, l, u, n_random, tighted_to_one=tighted_to_one, tighted_to_zero=tighted_to_zero)
        end

        return prune_left, prune_right
    end
end

function build_dual_branch_callback(
    A,
    N,
    f,
    sub_grad!;
    L=nothing,
    tightened=true,
    tighted_to_one=Dict{Int, Int}(),
    tighted_to_zero=Dict{Int, Int}(),
    processed_tightening_nodes=Ref(0),
)
    m, n = size(A)
    T = eltype(A)

    l = zeros(T, m)
    u = ones(T, m)
    fixed_mask = falses(m)

    fixed_to_zero = Vector{Int}(undef, m)
    fixed_to_one = Vector{Int}(undef, m)
    return function branch_callback(tree, node, vdix)
        if node.depth > N
            return false, false
        end
        processed_tightening_nodes[] += 1

        fill!(l, zero(T))
        fill!(u, one(T))
        fill!(fixed_mask, false)
        N_star = Int(N)

        # collecting current fixings
        int_vars = tree.root.problem.integer_variables
        for i in int_vars
            local_ub = get(node.local_bounds.upper_bounds, i, Inf)
            local_lb = get(node.local_bounds.lower_bounds, i, -Inf)
            l[i] = isfinite(local_lb) ? local_lb : zero(T)
            u[i] = isfinite(local_ub) ? local_ub : one(T)
            if isfinite(local_lb) || isfinite(local_ub)
                fixed_mask[i] = true
            end
            N_star = isfinite(local_lb) ? (N_star - 1) : N_star
        end

        y = node.active_set.x
        Y = L === nothing ? (A' * Diagonal(y) * A) : (L + A' * Diagonal(y) * A)
        λ, V = eigen(Y)
        if !isreal(λ[1])
            return false, false
        end
        λ_min = minimum(λ)

        # create dual problem
        opt = optimizer_with_attributes(Mosek.Optimizer, 
            MOI.Silent() => true, #!verbose,
        )
        #dual_sdp_model = Model(opt)
        dual_sdp_model = Model(dual_optimizer(opt))

        # Variables
        # λ: scalar variable
        JuMP.@variable(dual_sdp_model, λ)
        # Z: n×n positive semidefinite matrix
        JuMP.@variable(dual_sdp_model, Z[1:n, 1:n], PSD)
        # α: m-dim vector 
        JuMP.@variable(dual_sdp_model, α[1:m])
        # β: m-dim vector of non-negative variables
        JuMP.@variable(dual_sdp_model, β[1:m])


        JuMP.@constraint(dual_sdp_model, α >= 0)
        JuMP.@constraint(dual_sdp_model, β >= 0)

        if tightened
            ind_non_zero = findall(x -> x > 1e-8, min.(1 .- y, y))
            JuMP.@constraint(dual_sdp_model, α[ind_non_zero] == 0)
            JuMP.@constraint(dual_sdp_model, β[ind_non_zero] == 0)
        end

        # Constraint: Tr(Z) = 1
        # The trace is the sum of diagonal elements
        JuMP.@constraint(dual_sdp_model, sum(Z[i, i] for i in 1:n) == 1)

        for i in 1:m
            JuMP.@constraint(dual_sdp_model, λ - A[i, :]' * Z * A[i, :] -  α[i] + β[i] == 0)
        end

        # Objective: minimize λ
        if L === nothing
            @objective(dual_sdp_model, Min, λ*N - α' * l + β' * u)
        else
            @objective(dual_sdp_model, Min, λ*N - α' * l + β' * u + LinearAlgebra.dot(Z, L))
        end

        # solve dual problem
        optimize!(dual_sdp_model)

        # query solution
        status = termination_status(dual_sdp_model)
        UB = objective_value(dual_sdp_model)
        α_val = value.(α)
        β_val = value.(β)

        if UB > λ_min
            @show "UB > λ_min UB: $(UB) λ_min: $(λ_min)"
            return false, false
        end

        if UB < -tree.incumbent
            return false, false
        end

        zc = 0
        oc = 0
        for i in int_vars
            if fixed_mask[i] || i == vdix
                continue
            end
            right_bound = UB - α_val[i]
            left_bound = UB - β_val[i]
            if right_bound <= -tree.incumbent
                push!(node.local_bounds.upper_bounds, (i => 0.0))
                zc += 1
                fixed_to_zero[zc] = i
            elseif left_bound <= -tree.incumbent
                push!(node.local_bounds.lower_bounds, (i => 1.0))
                oc += 1
                fixed_to_one[oc] = i
            end
        end

        if (zc > 0 || oc > 0)
            fixed_to_zero_view = view(fixed_to_zero, 1:zc)
            fixed_to_one_view = view(fixed_to_one, 1:oc)
            @show fixed_to_zero_view
            @show fixed_to_one_view
        end
        zc > 0 ? push!(tighted_to_zero, node.id => zc) : nothing
        oc > 0 ? push!(tighted_to_one, node.id => oc) : nothing

        return false, false
    end
end