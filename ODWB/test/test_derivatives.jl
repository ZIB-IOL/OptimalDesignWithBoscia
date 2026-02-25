# Test if derivatives are correct
using FrankWolfe
using FiniteDifferences
using LinearAlgebra
using Random
using ODWB
using Test

seed = rand(UInt64)
@show seed
Random.seed!(seed)

"""
Check if the gradient using finite differences matches the grad! provided.
Copied from FrankWolfe package: https://github.com/ZIB-IOL/FrankWolfe.jl/blob/master/examples/plot_utils.jl
"""
function check_gradients(grad!, f, gradient, num_tests=10, tolerance=1.0e-5)
    for i in 1:num_tests
        random_point = rand(length(gradient))
        grad!(gradient, random_point)
        if norm(grad(central_fdm(5, 1), f, random_point)[1] - gradient) > tolerance
            @warn "There is a noticeable difference between the gradient provided and
            the gradient computed using finite differences.:\n$(norm(grad(central_fdm(5, 1), f, random_point)[1] - gradient))"
            return false
        end
    end
    return true
end


@testset "Derivative A-Opt" begin
    for dim in [20,50,80]
        n = Int(floor(dim/4))
        @show dim, n
        gradient = rand(dim)
        A, _, _, _, _ = ODWB.build_data(seed,dim, n, false, false)
        f, grad! = ODWB.build_a_criterion(A, false, μ=1e-2)

        @test check_gradients(grad!, f, gradient)
    end
end

@testset "Derivative D-Opt" begin
    for dim in [20,50,80]
        n = Int(floor(dim/4))
        @show dim, n
        gradient = rand(dim)
        A, _, _, _, _ = ODWB.build_data(seed,dim, n, false, false)
        f, grad! = ODWB.build_d_criterion(A, false, build_safe=true)

        @test check_gradients(grad!, f, gradient)
    end
end

@testset "Sanity check smoothing" begin
    for fusion in [false, true]
        for dim in [20, 50, 80]
            for mu in [0.2, 0.5, 1.0]
                for _ in 1:10
                    n = Int(floor(dim/10))
                    x = fusion ? rand(n) : rand(dim)
                    if fusion 
                        edges, potential_edges = ODWB.build_graph_connectivity_data(dim, dim, n, seed=seed, connected=fusion)
                        L = ODWB.graph_laplacian(dim, edges)
                        A = ODWB.potential_edges_incidence_matrix(dim, potential_edges)
                    else
                        A, _, _, _, _ = ODWB.build_data(seed, dim, n, false, false)
                        L = nothing
                    end
                    f, sub_grad!, generate_smoothing_function = ODWB.build_e_criterion(A, L=L)

                    f_mu, grad! = generate_smoothing_function(mu)

                    @test f(x) >= f_mu(x)

                    y = fusion ? rand(n) : rand(dim)
                    @test f_mu(mu * x + (1-mu) * y) <= mu * f_mu(x) + (1-mu) * f_mu(y)
                end
            end
        end
    end
end

@testset "Derivative E-opt" begin
    for dim in [20,50,80]
        for μ in [1e-2, 1e-1, 1.0]
            n = Int(floor(dim/4))
            @show dim, n
            gradient = rand(dim)
            A, _, _, _, _ = ODWB.build_data(seed, dim, n, false, false)
            f_orig, sub_grad!, generate_smoothing_function = ODWB.build_e_criterion(A)
            f_mu, grad! = generate_smoothing_function(μ)

            @test check_gradients(grad!, f_mu, gradient)
        end
    end
end 

@testset "Derivative A-Fusion" begin
    for dim in [20,50,80]
        n = Int(floor(dim/4))
        @show dim, n
        gradient = rand(dim)
        A, C, _, _, _ = ODWB.build_data(seed,dim, n, true, false)
        f, grad! = ODWB.build_a_criterion(A, true, C=C)

        @test check_gradients(grad!, f, gradient)
    end
end

@testset "Derivative D-Fusion" begin
    for dim in [20,50,80]
        n = Int(floor(dim/4))
        @show dim, n
        gradient = rand(dim)
        A, C, _, _, _ = ODWB.build_data(seed,dim, n, true, false)
        f, grad! = ODWB.build_d_criterion(A,true, C=C)

        @test check_gradients(grad!, f, gradient)
    end
end

@testset "Derivatives Custom BB solver" begin
    for p in 0:-0.5:-2
        @show p
        for dim in [20,50,80]
            n = Int(floor(dim/4))
            @show dim, n
            gradient = rand(dim)
            A, _, _, _, _ = ODWB.build_data(seed,dim, n, true, false)
            f, grad! = ODWB.build_matrix_means_objective(A,p)

            @test check_gradients(grad!, f, gradient)

            A, _, _, _, _ = ODWB.build_data(seed,dim, n, false, false)
            f, grad! = ODWB.build_matrix_means_objective(A,p)

            @test check_gradients(grad!, f, gradient)
        end
    end
end
