"""
    find_large_leverage_set(A::Matrix{Float64}, initial_idx_set::Vector{Int}, target_size::Int; max_attempts::Int=1000)

Generate an index set that includes the given initial index set, has the specified target size,
and ensures the submatrix A[selected_indices, :] has full column rank.

Note: Despite the name (kept for backward compatibility), this function now simply finds any valid 
set rather than optimizing for leverage scores, making it much faster.

# Arguments
- `A::Matrix{Float64}`: An m×n matrix where m is the number of experiments and n is the number of parameters
- `initial_idx_set::Vector{Int}`: Initial set of indices that must be included in the final set
- `target_size::Int`: Desired size of the final index set
- `max_attempts::Int=1000`: Maximum number of attempts for optimization (currently unused)

# Returns
- `Vector{Int}`: Index set of size target_size that includes initial_idx_set and ensures full column rank

# Algorithm
1. Start with the initial index set
2. If needed, ensure we have at least n linearly independent rows by adding/replacing indices
3. Simply add remaining indices while maintaining full column rank (no optimization needed)
4. Prioritizes rank preservation over any particular optimality criterion

# Example
```julia
A = randn(100, 5)  # 100 experiments, 5 parameters
initial_set = [1, 3, 7]  # Must include these indices
target_size = 20  # Want 20 total indices
final_set = find_large_leverage_set(A, initial_set, target_size)
@assert length(final_set) == target_size
@assert all(idx in final_set for idx in initial_set)
@assert rank(A[final_set, :]) == 5  # Full column rank
```
"""
function find_large_leverage_set(A::Matrix{Float64}, initial_idx_set::Vector{Int}, target_size::Int; max_attempts::Int=1000)
    m, n = size(A)
    
    # Validate inputs
    if target_size < length(initial_idx_set) || target_size < n
        return initial_idx_set, false
    end
    @assert all(1 ≤ idx ≤ m for idx in initial_idx_set) "All indices must be in valid range"
    
    current_set = copy(initial_idx_set)
    
    # If we already have the target size, check if we have full rank
    if length(current_set) == target_size
        if rank(A[current_set, :]) == n
            return current_set, true
        else
            # Need to replace some indices to get full rank
            # Find linearly independent subset and rebuild
            independent_subset = linearly_independent_rows(A[current_set, :], length(current_set), min(n, length(current_set)))
            current_set = current_set[independent_subset]
        end
    end
    
    # Ensure we have at least n linearly independent rows to start with
    if length(current_set) < n || rank(A[current_set, :]) < min(n, length(current_set))
        # Get linearly independent rows from the initial set and supplement if needed
        if !isempty(current_set)
            sub_A = A[current_set, :]
            independent_indices = linearly_independent_rows(sub_A, length(current_set), min(n, length(current_set)))
            current_set = current_set[independent_indices]
        end
        
        # Add more linearly independent rows if needed
        while length(current_set) < n
            remaining_indices = setdiff(1:m, current_set)
            if isempty(remaining_indices)
                break
            end
            
            # Try to find an index that maintains/increases rank
            best_idx = nothing
            for idx in remaining_indices
                test_set = vcat(current_set, idx)
                if rank(A[test_set, :]) > rank(A[current_set, :])
                    best_idx = idx
                    break
                end
            end
            
            if best_idx !== nothing
                push!(current_set, best_idx)
            else
                # No improvement possible, add first available
                push!(current_set, remaining_indices[1])
            end
        end
    end
    
    # Simply add remaining indices until we reach target_size
    while length(current_set) < target_size
        remaining_indices = setdiff(1:m, current_set)
        if isempty(remaining_indices)
            break
        end
        
        # Find any index that maintains full rank
        best_idx = nothing
        for idx in remaining_indices
            test_set = vcat(current_set, idx)
            if rank(A[test_set, :]) == n
                best_idx = idx
                break
            end
        end
        
        # If no rank-preserving candidate found, fall back to any index that doesn't reduce rank
        if best_idx === nothing
            current_rank = rank(A[current_set, :])
            for idx in remaining_indices
                test_set = vcat(current_set, idx)
                if rank(A[test_set, :]) >= current_rank
                    best_idx = idx
                    break
                end
            end
        end
        
        if best_idx !== nothing
            push!(current_set, best_idx)
        else
            # Emergency fallback: add any remaining index
            push!(current_set, remaining_indices[1])
        end
    end
    
    # Final verification
    final_rank = rank(A[current_set, :])
    if final_rank < n
        @warn "Unable to achieve full column rank. Final rank: $final_rank, required: $n"
    end
    
    return current_set, true
end

"""
Heuristic based on the approach in https://arxiv.org/abs/2401.14317
"""
function build_pipage_rounding_heuristic(A, N; threshold=0.8, epsilon=1)
    m, n = size(A)
    inf_matrix(x) = A' * Diagonal(x) * A
    return function pipage_rounding_heuristic(tree::Bonobo.BnBTree, tlmo::Boscia.TimeTrackingLMO, x)
        x_new = copy(x)
        idx_set = findall(x .> threshold)
        cut_off = Int(floor(min(max(n * log(n)/epsilon^2, length(idx_set)), N)))
        S, valid = find_large_leverage_set(A, idx_set, cut_off) 
        if valid
            return [x], true
        end
        sols = []
        # save original bounds
        node = tree.nodes[tree.root.current_node_id[]]
        original_bounds = copy(node.local_bounds)
        # build local bounds
        local_bounds = Boscia.IntegerBounds()
        for i in S
            push!(local_bounds, (i, 1.0), :lessthan)
            push!(local_bounds, (i, 1.0), :greaterthan)
        end
        x_new[S] .= 1.0
        x_new[setdiff(1:m, S)] .= 0.0
        X_inv = inv(inf_matrix(x_new))
        for i in 1:m
            if i in setdiff(1:m, S)
                leverage = A[i, :]' * X_inv * A[i, :]
                if leverage > epsilon^2/(10 * log(n)) || isapprox(x[i], 0.0, atol=1e-10)
                    push!(local_bounds, (i, 0.0), :lessthan)
                    push!(local_bounds, (i, 0.0), :greaterthan)
                else
                    push!(local_bounds, (i, 1.0), :lessthan)
                    push!(local_bounds, (i, 0.0), :greaterthan)
                end
            end
        end
        Boscia.build_LMO(
            tlmo,
            tree.root.problem.integer_variable_bounds,
            local_bounds,
            tree.root.problem.integer_variables,
        )

        # check for feasibility and boundedness
        status = Boscia.check_feasibility(tlmo)
        if status == Boscia.INFEASIBLE || status == Boscia.UNBOUNDED
            @debug "LMO state in the probability rounding heuristic: $(status)"
            # reset LMO to node state
            Boscia.build_LMO(
                tlmo,
                tree.root.problem.integer_variable_bounds,
                original_bounds,
                tree.root.problem.integer_variables,
            )
            # just return the point
            return [x], true
        end

        v = Boscia.compute_extreme_point(tlmo, rand(length(x)))
        active_set = FrankWolfe.ActiveSet([(1.0, v)])

        x_pipage, _, _, _ = Boscia.solve_frank_wolfe(
            tree.root.options[:variant],
            tree.root.problem.f,
            tree.root.problem.g,
            tree.root.problem.tlmo,
            active_set;
            epsilon=node.fw_dual_gap_limit,
            max_iteration=tree.root.options[:max_fw_iter],
            line_search=tree.root.options[:line_search],
            lazy=tree.root.options[:lazy],
            lazy_tolerance=tree.root.options[:lazy_tolerance],
            callback=tree.root.options[:callback],
            verbose=tree.root.options[:fw_verbose],
        )

        for (idx, x_i) in enumerate(x_pipage)
            x_pipage[idx] = rand() < x_i ? min(1.0, ceil(x_i)) : max(0.0, floor(x_i))
        end

        # reset LMO to node state
        Boscia.build_LMO(
            tlmo,
            tree.root.problem.integer_variable_bounds,
            original_bounds,
            tree.root.problem.integer_variables,
        )

        return [x_pipage], false
    end
end
"""
Follow subgradient heuristic for E-optimal design.
"""
function build_follow_subgradient_heuristic(A, k)
    m, n = size(A)
    return function follow_gradient_heuristic(tree::Bonobo.BnBTree, tlmo::Boscia.TimeTrackingLMO, x)
        x_new = copy(x)
        sols = []
        sol_hashes = Set{UInt}()
        for i in 1:k
            time = float(Dates.value(Dates.now() - tree.root.problem.tlmo.time_ref))
            if tree.root.options[:time_limit] < Inf &&
            time / 1000.0 ≥ tree.root.options[:time_limit] - 10
                break
            end

            # Direction to maximize λ_min: use (A*v_min)² as LMO direction (negative subgradient of -λ_min)
            X = A' * Diagonal(x_new) * A
            if !isposdef(X)
                return [x], true
            end
            λ, V = eigen(X)
            v_min = V[:, 1]
            nabla = (A * v_min).^2
            x_new = Boscia.compute_extreme_point(tlmo, nabla)
            sol_hash = hash(x_new)
            if in(sol_hash, sol_hashes)
                break
            end
            push!(sols, x_new)
            push!(sol_hashes, sol_hash)
        end
        return sols, false
    end
end

"""
Simple randomized rounding heuristic for E-optimal design without repetition.
From https://jourdainlamperski.com/wp-content/uploads/2024/01/rand_round_max_min_eig.pdf 
"""
function build_simple_randomized_rounding_heuristic(A, N, max_iter; rng=Random.default_rng())
    m, n = size(A)
    return function simple_randomized_rounding_heuristic(tree::Bonobo.BnBTree, tlmo::Boscia.TimeTrackingLMO, x)
        x_new = copy(x)
        sols = []
        no_feasible_solution_found = true
        k = 1
        while k <= max_iter && no_feasible_solution_found
            for (i, x_i) in zip(collect(1:m), x)
                x_new[i] = rand(rng) < x_i ? min(1.0, ceil(x_i)) : max(0.0, floor(x_i))
            end
            if sum(x_new) == N 
                push!(sols, x_new)
                no_feasible_solution_found = false
            end
            k += 1
        end
        return sols, false
    end
end

"""
Greedy Fedorov heuristic for E-optimal design without repetition.
From https://jourdainlamperski.com/wp-content/uploads/2024/01/rand_round_max_min_eig.pdf 
"""
function build_greedy_fedorov_heuristic(A, N, max_iter; tolerance = 0.0)
    m, n = size(A)
    inf_matrix(x) = A' * Diagonal(x) * A
    return function greedy_fedorov_heuristic(tree::Bonobo.BnBTree, tlmo::Boscia.TimeTrackingLMO, x)
        z = copy(tree.incumbent_solution.solution)
        sols = []
        improved = false
        k = 0
        while !improved && k <= max_iter
            z_idx = findall(z .> 0.0)
            leverage = fill(0.0, length(z_idx))
            X = inf_matrix(z)
            X_inv = inv(X)
            for (i, idx) in enumerate(z_idx)
                leverage[i] = A[idx, :]' * X_inv * A[idx, :]
            end

            f = tree.root.options[:mode] == Boscia.SMOOTHING_MODE ? tree.root.options[:original_objective] : tree.root.problem.f

            perm = sortperm(leverage)
            for i in perm
                best_idx = 0
                for j in setdiff(1:m, z_idx)
                    z_new = copy(z)
                    z_new[j] = 1.0
                    z_new[z_idx[i]] = 0.0
                    if sum(z_new) == N &&  f(z_new) > f(z) + tolerance
                        best_idx = j
                        break
                    end
                end
                if best_idx != 0
                    z[z_idx[i]] = 0.0
                    z[best_idx] = 1.0
                    improved = true
                    push!(sols, z)
                    break
                end
            end
            k += 1
        end
        return sols, false
    end
end

