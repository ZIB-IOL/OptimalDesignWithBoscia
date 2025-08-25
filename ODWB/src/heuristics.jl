# greedy heuristic with local swap steps
"""
    e_optimal_design(vectors::Matrix, N::Int; rng=Random.default_rng())

Heuristic for E-optimal design without repetition.
- `vectors` is an m×d matrix whose rows are candidate vectors.
- `N` is the number of design points to select.
Returns the set of indices chosen.
"""
function e_optimal_design(vectors::Matrix{Float64}, N::Int; rng=Random.default_rng())
    m, d = size(vectors)
    # Step 1: leverage scores (proxy for fractional x*)
    gram = vectors * transpose(vectors) # m×m, but we need leverage approx
    cov = transpose(vectors) * vectors
    M = pinv(cov) # use pseudo-inverse
    scores = vec(sum((vectors * M) .* vectors, dims=2))
    weights = scores ./ sum(scores) * N

    # Step 2: greedy rounding (initialize with top N by weight)
    perm = sortperm(weights, rev=true)
    chosen = Set(perm[1:N])

    # Step 3: local swap improvement (try replacing one point with another)
    function λmin_of_set(S)
        X = vectors[collect(S), :]
        return minimum(eigvals(transpose(X) * X))
    end

    improved = true
    while improved
        improved = false
        current_val = λmin_of_set(chosen)
        for i in chosen
            for j in setdiff(1:m, chosen)
                newset = union(setdiff(chosen, [i]), [j])
                newval = λmin_of_set(newset)
                if newval > current_val
                    chosen = newset
                    improved = true
                    break
                end
            end
            if improved
                break
            end
        end
    end

    return collect(chosen)
end

# Example usage
m, d, N = 50, 5, 10
A = randn(m, d) # candidate vectors
B = e_optimal_design(A, N)
println("Chosen indices: ", B)

"""
    e_optimal_design_pipage(vectors::Matrix, N::Int; rng=Random.default_rng())

Randomized pipage rounding heuristic for E-optimal design without repetition.

- `vectors`: m×d matrix, rows are candidate vectors.
- `N`: number of design points to select.
Returns indices of chosen design points.
"""
function e_optimal_design_pipage(vectors::Matrix{Float64}, N::Int; rng=Random.default_rng())
    m, d = size(vectors)

    # Step 1: Initialize fractional solution with leverage-score weights
    cov = transpose(vectors) * vectors
    M = pinv(cov)
    scores = vec(sum((vectors * M) .* vectors, dims=2))
    x = scores ./ sum(scores) * N  # fractional solution

    # Clip to [0,1] for safety
    x = clamp.(x, 0.0, 1.0)

    # Step 2: Randomized pipage rounding
    while any(y -> 0 < y < 1, x)
        # pick two fractional indices
        fractional = findall(y -> 0 < y < 1, x)
        if length(fractional) < 2
            break
        end
        i, j = rand(rng, fractional, 2; replace=false)

        # max possible movement
        δ_plus  = min(1 - x[i], x[j])   # move mass i←j
        δ_minus = min(x[i], 1 - x[j])   # move mass j←i

        # randomized decision
        p = δ_minus / (δ_plus + δ_minus)
        if rand(rng) < p
            # move in + direction
            x[i] += δ_plus
            x[j] -= δ_plus
        else
            # move in - direction
            x[i] -= δ_minus
            x[j] += δ_minus
        end
    end

    # Step 3: Extract set of chosen indices
    chosen = findall(y -> y ≥ 0.5, x)  # break ties by threshold
    # If not exactly N, adjust greedily
    if length(chosen) > N
        chosen = chosen[1:N]
    elseif length(chosen) < N
        remaining = setdiff(1:m, chosen)
        append!(chosen, remaining[1:(N - length(chosen))])
    end

    return chosen
end

# Example usage
m, d, N = 50, 5, 10
A = randn(m, d) # candidate vectors
B = e_optimal_design_pipage(A, N)
println("Chosen indices: ", B)



"""
    e_optimal_design_pipage_deterministic(vectors::Matrix, N::Int)

Deterministic pipage rounding heuristic for E-optimal design without repetition.

- `vectors`: m×d matrix, rows are candidate vectors.
- `N`: number of design points to select.
Returns indices of chosen design points.
"""
function e_optimal_design_pipage_deterministic(vectors::Matrix{Float64}, N::Int)
    m, d = size(vectors)

    # Step 1: Initialize fractional solution with leverage-score weights
    cov = transpose(vectors) * vectors
    M = pinv(cov)
    scores = vec(sum((vectors * M) .* vectors, dims=2))
    x = scores ./ sum(scores) * N  # fractional solution

    # Clip to [0,1] for safety
    x = clamp.(x, 0.0, 1.0)

    # Helper: objective surrogate (log-det of info matrix)
    function obj(xvec)
        M = transpose(vectors) * (Diagonal(xvec) * vectors)
        # add small ridge for numerical stability
        return logdet(M + 1e-8 * I)
    end

    # Step 2: Deterministic pipage rounding
    while any(y -> 0 < y < 1, x)
        # pick two fractional indices
        fractional = findall(y -> 0 < y < 1, x)
        if length(fractional) < 2
            break
        end
        i, j = fractional[1:2]  # pick first two (could randomize)

        # max possible movements
        δ_plus  = min(1 - x[i], x[j])   # move mass i←j
        δ_minus = min(x[i], 1 - x[j])   # move mass j←i

        # evaluate objectives for both moves
        x_plus = copy(x); x_plus[i] += δ_plus; x_plus[j] -= δ_plus
        x_minus = copy(x); x_minus[i] -= δ_minus; x_minus[j] += δ_minus

        if obj(x_plus) ≥ obj(x_minus)
            x = x_plus
        else
            x = x_minus
        end
    end

    # Step 3: Extract set of chosen indices
    chosen = findall(y -> y ≥ 0.5, x)
    # Fix if necessary
    if length(chosen) > N
        chosen = chosen[1:N]
    elseif length(chosen) < N
        remaining = setdiff(1:m, chosen)
        append!(chosen, remaining[1:(N - length(chosen))])
    end

    return chosen
end

# Example usage
m, d, N = 50, 5, 10
A = randn(m, d)
B = e_optimal_design_pipage_deterministic(A, N)
println("Chosen indices: ", B)


