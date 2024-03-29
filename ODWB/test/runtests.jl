using ODWB
using Boscia
using Pajarito
using SCIP
using Hypatia
using FrankWolfe
using Random
using Test

seed = rand(UInt64)
#seed = 0xad2cba3722a98b62
@show seed
Random.seed!(seed)


include("test_derivatives.jl")
include("fusion_co-bnb.jl")

dimensions = [20,30]
facs = [10,4]
time_limit = 300
verbose = false

@testset "Boscia" begin
   @testset "A Optimal" begin
        println("A-Optimal Test")
        for m in dimensions 
            for k in facs 
                n = Int(floor(m/k))
                @show m, n
                x_ind, _ = ODWB.solve_opt(seed, m ,n, time_limit, "A", false; write = false, verbose=verbose)
                x_corr, _ = ODWB.solve_opt(seed, m, n, time_limit, "A", true; write = false, verbose=verbose)

                A, _, N, ub = ODWB.build_data(seed, m , n, false, false)
                check_lmo = ODWB.build_blmo(m, N, ub)
                @test Boscia.is_linear_feasible(check_lmo, x_ind)
                @test isapprox(sum(x_ind), N; atol=1e-4, rtol=1e-2)

                A, _, N, ub = ODWB.build_data(seed, m , n, false, true)
                check_lmo = ODWB.build_blmo(m, N, ub)
                @test Boscia.is_linear_feasible(check_lmo, x_corr)
                @test isapprox(sum(x_corr), N; atol=1e-4, rtol=1e-2)
            end
        end
    end 

   @testset "D Optimal" begin
    println("D-Optimal Test")
        for m in dimensions 
            for k in facs 
                n = Int(floor(m/k))
                x_ind, _ = ODWB.solve_opt(seed, m ,n, time_limit, "D", false; write = false, verbose=verbose)
                x_corr, _ = ODWB.solve_opt(seed, m, n, time_limit, "D", true; write = false, verbose=verbose )
                A, _, N, ub = ODWB.build_data(seed, m , n, false, false)
                check_lmo = ODWB.build_blmo(m, N, ub)
                @test Boscia.is_linear_feasible(check_lmo, x_ind)

                A, _, N, ub = ODWB.build_data(seed, m , n, false, true)
                check_lmo = ODWB.build_blmo(m, N, ub)
                @test Boscia.is_linear_feasible(check_lmo, x_corr)
                @test isapprox(sum(x_corr), N; atol=1e-4, rtol=1e-2)
            end
        end
    end 

    @testset "A Fusion" begin
        println("A-Fusion Test")
        for m in dimensions 
            for k in facs 
                n = Int(floor(m/k))
                x_ind, _ = ODWB.solve_opt(seed, m ,n, time_limit, "AF", false; write = false, verbose=verbose)
                x_corr, _ = ODWB.solve_opt(seed, m, n, time_limit, "AF", true; write = false, verbose=verbose)

                A, _, N, ub = ODWB.build_data(seed, m , n, true, false)
                check_lmo = ODWB.build_blmo(m, N, ub)
                @test Boscia.is_linear_feasible(check_lmo, x_ind)
                @test isapprox(sum(x_ind), N; atol=1e-4, rtol=1e-2)

                A, _, N, ub = ODWB.build_data(seed, m , n, true, true)
                check_lmo = ODWB.build_blmo(m, N, ub)
                @test Boscia.is_linear_feasible(check_lmo, x_corr)
                @test isapprox(sum(x_corr), N; atol=1e-4, rtol=1e-2)
            end
        end
    end

    @testset "D Fusion" begin
        println("D-Fusion Test")
        for m in dimensions 
            for k in facs 
                n = Int(floor(m/k))
                x_ind, _ = ODWB.solve_opt(seed, m ,n, time_limit, "DF", false; write = false, verbose=verbose)
                x_corr, _ = ODWB.solve_opt(seed, m, n, time_limit, "DF", true; write = false, verbose=verbose)

                A, _, N, ub = ODWB.build_data(seed, m , n, true, false)
                check_lmo = ODWB.build_blmo(m, N, ub)
                @test Boscia.is_linear_feasible(check_lmo, x_ind)
                @test isapprox(sum(x_ind), N; atol=1e-4, rtol=1e-2)

                A, _, N, ub = ODWB.build_data(seed, m , n, true, true)
                check_lmo = ODWB.build_blmo(m, N, ub)
                @test Boscia.is_linear_feasible(check_lmo, x_corr)
                @test isapprox(sum(x_corr), N; atol=1e-4, rtol=1e-2)
            end
        end
    end 
end


@testset "Custom BB" begin
    @testset "A Optimal" begin
        for m in dimensions 
            for k in facs 
                n = Int(floor(m/k))
                println("\n")
                @show m, n
                x_ind, time_ind, _, _, _ = @timed ODWB.solve_opt_custom(seed, m ,n, time_limit, "A", false; write = false, verbose=verbose)
                @show time_ind
                x_corr, time_corr, _, _, _ = @timed ODWB.solve_opt_custom(seed, m, n, time_limit, "A", true; write = false, verbose=verbose)
                @show time_corr

                A, _, N, ub = ODWB.build_data(seed, m , n, false, false)
                check_lmo = ODWB.build_blmo(m, N, ub)
                @test Boscia.is_linear_feasible(check_lmo, x_ind)
                @test isapprox(sum(x_ind), N; atol=1e-4, rtol=1e-2)

                A, _, N, ub = ODWB.build_data(seed, m , n, false, true)
                check_lmo = ODWB.build_blmo(m, N, ub)
                @test Boscia.is_linear_feasible(check_lmo, x_corr)
                @test isapprox(sum(x_corr), N; atol=1e-4, rtol=1e-2)
             end
        end
    end 

   @testset "D Optimal" begin
        for m in dimensions 
            for k in facs 
                n = Int(floor(m/k))
                x_ind, time_ind, _,_,_ = @timed ODWB.solve_opt_custom(seed, m ,n, time_limit, "D", false; write = false, verbose=verbose)
                x_corr, time_corr, _,_,_ = @timed ODWB.solve_opt_custom(seed, m, n, time_limit, "D", true; write = false, verbose=verbose)

                A, _, N, ub = ODWB.build_data(seed, m , n, false, false)
                check_lmo = ODWB.build_blmo(m, N, ub)
                @test Boscia.is_linear_feasible(check_lmo, x_ind)
                @test isapprox(sum(x_ind), N; atol=1e-4, rtol=1e-2)

                A, _, N, ub = ODWB.build_data(seed, m , n, false, true)
                check_lmo = ODWB.build_blmo(m, N, ub)
                @test Boscia.is_linear_feasible(check_lmo, x_corr)
                @test isapprox(sum(x_corr), N; atol=1e-4, rtol=1e-2)
            end
        end
    end 
end 


@testset "SCIP" begin
    @testset "A Fusion" begin
        for m in dimensions 
            for k in facs 
                n = Int(floor(m/k))
                x_ind = ODWB.solve_opt_scip(seed, m ,n, time_limit, "AF", false; write = false, verbose=verbose)
                x_corr = ODWB.solve_opt_scip(seed, m, n, time_limit, "AF", true; write = false, verbose=verbose)

                A, _, N, ub = ODWB.build_data(seed, m , n, true, false)
                check_lmo = ODWB.build_blmo(m, N, ub)
                @test Boscia.is_linear_feasible(check_lmo, x_ind)
                @test isapprox(sum(x_ind), N; atol=1e-4, rtol=1e-2)

                A, _, N, ub = ODWB.build_data(seed, m , n, true, true)
                check_lmo = ODWB.build_blmo(m, N, ub)
                @test Boscia.is_linear_feasible(check_lmo, x_corr)
                @test isapprox(sum(x_corr), N; atol=1e-4, rtol=1e-2)
            end
        end
    end

    @testset "D Fusion" begin
        for m in dimensions 
            for k in facs 
                n = Int(floor(m/k))
                x_ind = ODWB.solve_opt_scip(seed, m ,n, time_limit, "DF", false; write = false, verbose=verbose)
                x_corr = ODWB.solve_opt_scip(seed, m, n, time_limit, "DF", true; write = false, verbose=verbose)

                A, _, N, ub = ODWB.build_data(seed, m , n, true, false)
                check_lmo = ODWB.build_blmo(m, N, ub)
                @test Boscia.is_linear_feasible(check_lmo, x_ind)
                @test isapprox(sum(x_ind), N; atol=1e-4, rtol=1e-2)

                A, _, N, ub = ODWB.build_data(seed, m , n, true, true)
                check_lmo = ODWB.build_blmo(m, N, ub)
                @test Boscia.is_linear_feasible(check_lmo, x_corr)
                @test isapprox(sum(x_corr), N; atol=1e-4, rtol=1e-2)
            end
        end
    end
end


@testset "Pajarito" begin
    @testset "A Optimal" begin
        for m in dimensions 
            for k in facs 
                n = Int(floor(m/k))
                x_ind = ODWB.solve_opt_pajarito(seed, m ,n, time_limit, "A", false; write = false, verbose=verbose)
                x_corr = ODWB.solve_opt_pajarito(seed, m, n, time_limit, "A", true; write = false, verbose=verbose)

                A, _, N, ub = ODWB.build_data(seed, m , n, false, false)
                check_lmo = ODWB.build_blmo(m, N, ub)
                @test Boscia.is_linear_feasible(check_lmo, x_ind)
                @test isapprox(sum(x_ind), N; atol=1e-4, rtol=1e-2)

                A, _, N, ub = ODWB.build_data(seed, m , n, false, true)
                check_lmo = ODWB.build_blmo(m, N, ub)
                @test Boscia.is_linear_feasible(check_lmo, x_corr)
                @test isapprox(sum(x_corr), N; atol=1e-4, rtol=1e-2)
            end
        end
    end

    @testset "D Optimal" begin
        for m in dimensions 
            for k in facs 
                n = Int(floor(m/k))
                x_ind= ODWB.solve_opt_pajarito(seed, m ,n, time_limit, "D", false; write = false, verbose=verbose)
                x_corr = ODWB.solve_opt_pajarito(seed, m, n, time_limit, "D", true; write = false, verbose=verbose)

                A, _, N, ub = ODWB.build_data(seed, m , n, false, false)
                check_lmo = ODWB.build_blmo(m, N, ub)
                @test Boscia.is_linear_feasible(check_lmo, x_ind)
                @test isapprox(sum(x_ind), N; atol=1e-4, rtol=1e-2)

                A, _, N, ub = ODWB.build_data(seed, m , n, false, true)
                check_lmo = ODWB.build_blmo(m, N, ub)
                @test Boscia.is_linear_feasible(check_lmo, x_corr)
                @test isapprox(sum(x_corr), N; atol=1e-4, rtol=1e-2)
            end
        end
    end

    @testset "A Fusion" begin
        for m in dimensions 
            for k in facs 
                n = Int(floor(m/k))
                x_ind = ODWB.solve_opt_pajarito(seed, m ,n, time_limit, "AF", false; write = false, verbose=verbose)
                x_corr = ODWB.solve_opt_pajarito(seed, m, n, time_limit, "AF", true; write = false, verbose=verbose)

                A, _, N, ub = ODWB.build_data(seed, m , n, true, false)
                check_lmo = ODWB.build_blmo(m, N, ub)
                @test Boscia.is_linear_feasible(check_lmo, x_ind)
                @test isapprox(sum(x_ind), N; atol=1e-4, rtol=1e-2)

                A, _, N, ub = ODWB.build_data(seed, m , n, true, true)
                check_lmo = ODWB.build_blmo(m, N, ub)
                @test Boscia.is_linear_feasible(check_lmo, x_corr)
                @test isapprox(sum(x_corr), N; atol=1e-4, rtol=1e-2)
            end
        end
    end

    @testset "D Fusion" begin
        for m in dimensions 
            for k in facs 
                n = Int(floor(m/k))
                x_ind = ODWB.solve_opt_pajarito(seed, m ,n, time_limit, "DF", false; write = false, verbose=verbose)
                x_corr = ODWB.solve_opt_pajarito(seed, m, n, time_limit, "DF", true; write = false, verbose=verbose)

                A, _, N, ub = ODWB.build_data(seed, m , n, true, false)
                check_lmo = ODWB.build_blmo(m, N, ub)
                @test Boscia.is_linear_feasible(check_lmo, x_ind)
                @test isapprox(sum(x_ind), N; atol=1e-4, rtol=1e-2)

                A, _, N, ub = ODWB.build_data(seed, m , n, true, true)
                check_lmo = ODWB.build_blmo(m, N, ub)
                @test Boscia.is_linear_feasible(check_lmo, x_corr)
                @test isapprox(sum(x_corr), N; atol=1e-4, rtol=1e-2)
            end
        end
    end
end 
