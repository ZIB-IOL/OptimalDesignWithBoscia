# Script for running the experiments
using ODWB
using Printf
using Test
using DataFrames
using CSV


# For debugging
#=
ENV["MODE"] = "Boscia"
ENV["CRITERION"] = "AGC"
ENV["TYPE"] = "IND"
ENV["SEED"] = "1"
ENV["OPTION"] = "exclusion_criterion"
ENV["N"] = "one"
ENV["DIMENSION"] = "80"
=#

"""
DON'T FORGET TO ADD e AGAIN ONCE YOU ARE DONE DEBUGGING!!
"""
mode = ENV["MODE"]
criterion = ENV["CRITERION"]
type = ENV["TYPE"]
m = parse(Int, ENV["DIMENSION"])
corr = if type == "IND"
    false 
elseif type == "CORR"
    true
else 
    error("Type not found")
end
seed = parse(Int, ENV["SEED"])
option = ENV["OPTION"]
N_construct = ENV["N"]
ratio_para = criterion in ["E", "EF", "AGC"] ? [1] : [4,10]
time_limit = 3600 # one hour time limit
seeds = seed == 0 ? collect(1:5) : [seed]

if option == "mu_testing"
    starts = [m/50, exp10(-200/m)]
    decays = [1.0, 0.9, 0.7]
elseif criterion == "AGC"
    starts = [m/200]
    decays = corr ? m in [80, 100] ? [0.9] : [0.7] : [0.9]
else
    starts = N_construct == "rank_deficient" && !corr ? [m/5] : [m/10] 
    decays = N_construct == "log" ? [0.9] : N_construct == "rank_deficient" ? [0.7] : [0.8]
end


@show criterion, mode, corr

if !(criterion in ["A", "D", "DF", "AF", "E", "EF", "AGC"])
    error("Invalid criterion!")
end
for k in ratio_para
    n = if criterion== "AGC"
        Int(floor(m/3))
    else
        k == 1 ? Int(floor(sqrt(m))) : Int(floor(m/k))
    end
    N = if criterion == "AGC"
        Int(floor(m/2))
    elseif N_construct == "one"
        Int(floor(1.5 * n))
    elseif N_construct == "log"
        Int(floor(1.5 * n * log(n)))
    elseif N_construct == "rank_deficient"
        Int(floor(3n/4))
    elseif N_construct == "nothing"
        -Inf
    else
        error("Invalid N_construct!")
    end
    for seed in seeds
        for decay in decays
            for start in starts
                if decay != 1.0 && start == exp10(-200/m)
                    continue
                end
                min = if criterion == "AGC"
                   N_construct == "rank_deficient" ? exp10(-100/m) : m in [80, 100] ? exp10(-300/m) : exp10(-400/m)
                else
                   exp10(-20/m)
                end
                @show m, n, N, seed, decay, start, min
                try
                    if mode == "Boscia"
                        ODWB.solve_opt(
                            seed, 
                            m, 
                            n, 
                            time_limit, 
                            criterion, 
                            corr, 
                            N=N, 
                            smoothing_start=start,
                            smoothing_decay=decay,
                            smoothing_min=min,#exp10(-200/m), exp10(-400/m) for AGC
                            use_exclusion_criterion=option in ["exclusion_criterion", "exclusion_criterion_random", "exclusion_criterion_tighter_tol", "dual_exclusion_criterion"], 
                            use_dual_exclusion_criterion=option == "dual_exclusion_criterion",
                            use_heuristics=option == "all_heuristics", 
                            use_follow_subgradient_heu=option == "follow_subgradient", 
                            use_pipage_heu=option == "pipage_rounding", 
                            use_sr_rounding_heu=option == "sr_rounding", 
                            use_fedorov_heu=option == "fedorov", 
                            options_run=option != "baseline", 
                            mu_testing=option == "mu_testing",
                            connected = criterion == "AGC" ? corr : true,
                            tightened = option in ["tightened", "tightened_scaled"],
                            scale = option == "tightened_scaled" ? 0.5 : Inf,
                            fw_verbose = false,
                            n_random = option == "exclusion_criterion_random" ? 10 : 0,
                            start_epsilon = option == "exclusion_criterion_tighter_tol" ? 1e-4 : 1e-2,
                            min_epsilon = option == "exclusion_criterion_tighter_tol" ? 1e-7 : 1e-6)
                    elseif mode == "SCIP"
                        if criterion in ["A", "D", "E", "EF"]
                        error("SCIP OA does not work with the $(criterion)-optimal problems!")
                        end
                        ODWB.solve_opt_scip(seed, m, n, time_limit, criterion, corr, N=N)
                    elseif mode in ["SCIPSDP", "SCIPSDP_oa", "SCIPSDP_bnb"]
                        if criterion in ["A", "D", "AF", "DF"]
                        error("SCIP SDP does not work with the $(criterion)-optimal problems!")
                        end
                        solving_mode = if mode  in ["SCIPSDP", "SCIPSDP_bnb"]
                            :bnb
                        elseif mode == "SCIPSDP_oa"
                            :oa
                        else
                            error("Invalid mode!")
                        end
                        ODWB.solve_opt_scip_sdp(
                            seed, 
                            m, 
                            n, 
                            time_limit, 
                            criterion, 
                            corr, 
                            N=N, 
                            scip_sdp_mode=solving_mode,
                            connected = criterion == "AGC" ? corr : true,
                            tightened = option == "tightened",
                            scale = option == "tightened_scaled" ? 0.5 : Inf) #scip_sdp_mode=option == "oa" ? :oa : :bnb
                    elseif mode == "Pajarito"
                        ODWB.solve_opt_pajarito(seed, m, n, time_limit, criterion, corr, integer_data=integer_data, N=N)
                    elseif mode == "Custom"
                        if criterion in ["E", "EF", "AGC"]
                            error("Co-BnB does not work with the $(criterion)-optimal problems!")
                        end
                        ODWB.solve_opt_custom(seed, m, n, time_limit, criterion, corr, N=N)
                    elseif mode == "SOCP"
                        if criterion in ["E", "EF", "AGC"]
                            error("SOCP does not work with the $(criterion)-optimal problems!")
                        end
                        ODWB.solve_opt_socp(seed, m, n, time_limit, criterion, corr, N=N)
                    else 
                        error("Invalid mode!")
                    end
                catch e
                    println(e)
                    error_file = criterion * "_opt_" * mode * "_" * type * "_" * "_" * option * ".txt" 
                    open(error_file,"a") do io
                        println(io, seed, " ", m, " ", N, " ", mode, " ", start, " ", decay, " : ", e)
                    end
                end
            end
        end
    end
end
