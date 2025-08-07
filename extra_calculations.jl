using Boscia
using LinearAlgebra

# M < (n-1)^(1/n)^(n-1) / n^(1/n)
for i in 10:10:100 #100:100:10000
    a = ((i-1)^(1/i))^(i-1)
    b = i^(1/i)
    m = a/b
    M = floor(m)
    c = ((i-1)/i)^(i-1)*M^(-1*(i-1))
    d = M/i^(i-1)   
    δ = c - d
    @show i, m, a, b, M, δ, c, d
end
println("\n What if we fix M and let only n vary?")
for M in 2:1:5
    @show M
    M = Float64(M)
    for i in 10:10:100
        c = ((i-1)/(i*M))^(i-1)
        #d = M/i^(i-1)   
        d = M
        δ = c - d
        @show i, δ, c, d
    end
    println("\n")
end
println("\n Correction of M: M < (n-1/n)^(n-1/n)\n")
for i in 10:10:100
    M = ((i-1)/i)^((i-1)/i)
    @show i, M
end
println("\n How is the distance?\n")
m = 25
M = 10
n = Int(floor(sqrt(m)))
N = Int(floor(1.5 * n))
A = rand(collect(1:M),m, n)
function min_distance(A, N)
    m, n = size(A)
    inf_matrix(x) = A' * Diagonal(x) * A
    f(x) = maximum(eigvals((-1) * inf_matrix(x)))
    distance = Inf
    min, min_sol = Boscia.min_via_enum_prob_simplex(f, m, N)
    solutions = Iterators.product(fill(0:1, m)...)
    for solution in solutions
        sol_vec = collect(solution)
        if sum(sol_vec) != N
            continue
        end
        value = f(sol_vec)
        if isapprox(value, min, atol=1e-6)
            continue
        end
        if value - min < distance
            distance = value - min
        end
    end
    return distance
end

distance = min_distance(A, N)
@show distance
@show det(A' * A), minimum(eigvals(A' * A)), maximum(eigvals(A' * A)), LinearAlgebra.tr(A' * A)
@show maximum(eigvals(A' * A))/minimum(eigvals(A' * A))
@show distance/minimum(eigvals(A' * A))
@show nothing
@show 1/det(A' * A), 1/LinearAlgebra.tr(A' * A), LinearAlgebra.tr(A' * A)/det(A' * A)
@show 1/det(A' * A) * minimum(eigvals(A' * A))
@show 1/LinearAlgebra.tr(A' * A) * minimum(eigvals(A' * A))
@show 1/LinearAlgebra.tr(A' * A) * maximum(eigvals(A' * A))/minimum(eigvals(A' * A))

count = 0
for M in 5:1:15
    @show M
    for _ in 1:10
        A = rand(collect(-M:M),m, n)
        distance = min_distance(A, N)
        @show distance
        if distance < 1/LinearAlgebra.tr(A' * A) * (1/(2*M)) * minimum(eigvals(A' * A))
            @show distance, 1/LinearAlgebra.tr(A' * A) * (1/(2*M)) * minimum(eigvals(A' * A))
            global count += 1
        end
    end
    println("\n")
end
@show count
