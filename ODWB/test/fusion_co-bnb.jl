using ODWB
using Boscia
using Pajarito
using SCIP
using Hypatia
using FrankWolfe
using Random
using Test

#seed = rand(UInt64)
#seed = 0xad2cba3722a98b62
#@show seed
#Random.seed!(seed)

seed =1
Random.seed!(seed)

dimensions = [20,30] # 30,40
facs = [10]#[4,6]
time_limit = 300 #1200
verbose = false

@testset "A Fusion Independent" begin
    for m in dimensions 
        for k in facs 
            n = Int(floor(m/k))
            # independent data
            x_b_ind, _ = ODWB.solve_opt(seed, m ,n, time_limit, "AF", false; write = false, verbose=verbose)
            x_co_ind = ODWB.solve_opt_custom(seed, m, n, time_limit, "AF", false; write=false, verbose=verbose)

            # Testing
            A, C, N, ub, _ = ODWB.build_data(seed, m , n, true, false)
            check_lmo = ODWB.build_blmo(m, N, ub)
            f, _ = ODWB.build_a_criterion(A, true, C=C)

            @show f(x_b_ind)
            @show f(x_co_ind[1:m])

            @test Boscia.is_linear_feasible(check_lmo, x_co_ind[1:m])
            #@test f(x_b_ind) ≤ f(x_co_ind[1:m])
        end
    end
end

@testset "D Fusion Independent" begin
    for m in dimensions 
        for k in facs 
            n = Int(floor(m/k))
            # independent data
            x_b_ind, _ = ODWB.solve_opt(seed, m ,n, time_limit, "DF", false; write = false, verbose=verbose)
            x_co_ind = ODWB.solve_opt_custom(seed, m, n, time_limit, "DF", false; write=false, verbose=verbose)

            # Testing
            A, C, N, ub, _ = ODWB.build_data(seed, m , n, true, false)
            check_lmo = ODWB.build_blmo(m, N, ub)
            f, _ = ODWB.build_d_criterion(A, true, C=C)

            @show f(x_b_ind)
            @show f(x_co_ind[1:m])

            @test Boscia.is_linear_feasible(check_lmo, x_co_ind[1:m])
            #@test f(x_b_ind) ≤ f(x_co_ind[1:m])
        end
    end
end 

@testset "A Fusion Correlated" begin
    for m in dimensions 
        for k in facs 
            n = Int(floor(m/k))
        
            x_b_corr, _ = ODWB.solve_opt(seed, m, n, time_limit, "AF", true; write = false, verbose=verbose)
            x_co_corr = ODWB.solve_opt_custom(seed, m, n, time_limit, "AF", true; write=false, verbose=verbose)

            # Testing
            A, C, N, ub, _ = ODWB.build_data(seed, m , n, true, true)
            check_lmo = ODWB.build_blmo(m, N, ub)
            f, _ = ODWB.build_a_criterion(A, true, C=C)

            @show f(x_b_corr)
            @show f(x_co_corr[1:m])

            @test Boscia.is_linear_feasible(check_lmo, x_co_corr[1:m])
            #@test f(x_b_corr) ≤ f(x_co_corr[1:m])
        end
    end
end

@testset "D Fusion Correlated" begin
    for m in dimensions 
        for k in facs 
            n = Int(floor(m/k))
            
            x_b_corr, _ = ODWB.solve_opt(seed, m, n, time_limit, "DF", true; write = false, verbose=verbose)
            x_co_corr = ODWB.solve_opt_custom(seed, m, n, time_limit, "DF", true; write=false, verbose=verbose)

            # Testing
            A, C, N, ub, _ = ODWB.build_data(seed, m , n, true, true)
            check_lmo = ODWB.build_blmo(m, N, ub)
            f, _ = ODWB.build_d_criterion(A, true, C=C)

            @show f(x_b_corr)
            @show f(x_co_corr[1:m])

            @test Boscia.is_linear_feasible(check_lmo, x_co_corr[1:m])
           # @test f(x_b_corr) ≤ f(x_co_corr[1:m])
        end
    end
end 
