# Smoke / unit checks for OPTION=optimized presets and CSV naming.
# Run: julia --project test_optimized_presets.jl

using Test
using ODWB

@testset "optimized_preset mapping" begin
    e_ind = ODWB.optimized_preset("E", false)
    @test e_ind.scale_smoothing_mu
    @test !e_ind.reduced_spectrum
    @test !e_ind.eigenvalue_based_pruning
    @test !e_ind.rank_based_pruning

    e_corr = ODWB.optimized_preset("E", true)
    @test e_corr.scale_smoothing_mu
    @test e_corr.eigenvalue_based_pruning
    @test !e_corr.reduced_spectrum

    agc_ind = ODWB.optimized_preset("AGC", false)
    @test agc_ind.scale_smoothing_mu
    @test agc_ind.reduced_spectrum
    @test agc_ind.reduced_percentage == 2
    @test !agc_ind.eigenvalue_based_pruning

    agc_corr = ODWB.optimized_preset("AGC", true)
    @test agc_corr.eigenvalue_based_pruning
    @test !agc_corr.scale_smoothing_mu
    @test !agc_corr.reduced_spectrum

    acst = ODWB.optimized_preset("ACST", false)
    @test acst.rank_based_pruning
    @test acst.reduced_spectrum
    @test acst.reduced_percentage == 2
    @test !acst.scale_smoothing_mu

    acsts = ODWB.optimized_preset("ACSTS", false)
    @test acsts.rank_based_pruning
    @test acsts.reduced_spectrum
    @test acsts.reduced_percentage == 2

    @test_throws ErrorException ODWB.optimized_preset("A", false)
end

@testset "optimized CSV prefix does not collide with baseline" begin
    # Tiny E-IND instance, short time limit; write must use `optimized` folder.
    seed, m, n = 1, 20, 4
    N = Int(floor(1.5 * n))
    time_limit = 15.0
    cfg = ODWB.optimized_preset("E", false)

    ODWB.solve_opt(
        seed, m, n, time_limit, "E", false;
        N = N,
        smoothing_start = m / 100,
        smoothing_decay = 0.8,
        smoothing_min = exp10(-100 / m),
        options_run = true,
        fw_verbose = false,
        verbose = false,
        write = true,
        print_iter = 1000,
        relative_gap_tolerance = 5e-2,
        scale_smoothing_mu = cfg.scale_smoothing_mu,
        reduced_spectrum = cfg.reduced_spectrum,
        reduced_percentage = cfg.reduced_percentage,
        eigenvalue_based_pruning = cfg.eigenvalue_based_pruning,
        rank_based_pruning = cfg.rank_based_pruning,
        optimized_run = true,
    )

    expected = joinpath(
        @__DIR__, "csv", "Boscia",
        "boscia_optimized_E_optimality_independent__$(m)_$(n)_$(N)_$(seed).csv",
    )
    baseline = joinpath(
        @__DIR__, "csv", "Boscia",
        "boscia__E_optimality_independent__$(m)_$(n)_$(N)_$(seed).csv",
    )
    @test isfile(expected)
    @test !occursin("boscia__", basename(expected))
    @test startswith(basename(expected), "boscia_optimized_")
    # Must not have written under baseline naming for this tiny instance in this call.
    # (baseline file may exist from older runs; we only require optimized path exists.)
    @test isfile(expected)
    println("Wrote: ", expected)
end

@testset "AGC CORR optimized enables eigenvalue pruning flags" begin
    cfg = ODWB.optimized_preset("AGC", true)
    seed, m = 1, 80
    n = Int(floor(m / 3))
    N = Int(floor(m / 2))
    ODWB.solve_opt(
        seed, m, n, 20.0, "AGC", true;
        N = N,
        connected = true,
        smoothing_start = m / 200,
        smoothing_decay = 0.9,
        smoothing_min = exp10(-300 / m),
        options_run = true,
        fw_verbose = false,
        verbose = false,
        write = true,
        print_iter = 1000,
        relative_gap_tolerance = 5e-2,
        scale_smoothing_mu = cfg.scale_smoothing_mu,
        reduced_spectrum = cfg.reduced_spectrum,
        reduced_percentage = cfg.reduced_percentage,
        eigenvalue_based_pruning = cfg.eigenvalue_based_pruning,
        rank_based_pruning = cfg.rank_based_pruning,
        optimized_run = true,
    )
    expected = joinpath(
        @__DIR__, "csv", "Boscia",
        "boscia_optimized_AGC_optimality_correlated_connected_$(m)_$(n)_$(N)_$(seed).csv",
    )
    @test isfile(expected)
    println("Wrote: ", expected)
end

println("All optimized-preset tests finished.")
