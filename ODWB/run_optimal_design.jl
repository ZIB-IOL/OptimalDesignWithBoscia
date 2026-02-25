# Script for running the experiments
using ODWB
using Printf
using Test
using DataFrames
using CSV

#=
# For debugging
ENV["MODE"] = "Boscia"
ENV["CRITERION"] = "E"
ENV["TYPE"] = "IND"
ENV["INTEGER_DATA"] = "false"
ENV["SEED"] = "1"
ENV["OPTION"] = "mu_testing"
ENV["N"] = "log"
ENV["DIMENSION"] = "50"
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
integer_data = parse(Bool, ENV["INTEGER_DATA"])
seed = parse(Int, ENV["SEED"])
option = ENV["OPTION"]
N_construct = ENV["N"]
ratio_para = criterion in ["E", "EF"] ? [1] : [4,10]
time_limit = 3600 # one hour time limit
seeds = seed == 0 ? collect(1:5) : [seed]

if option == "mu_testing"
    starts = [m/50, exp10(-200/m)]
    decays = [1.0, 0.9, 0.7]
else
    starts = [m/50]
    decays = [0.7]
end


@show criterion, mode, corr

if !(criterion in ["A", "D", "DF", "AF", "E", "EF"])
    error("Invalid criterion!")
end
for k in ratio_para
    n = k == 1 ? Int(floor(sqrt(m))) : Int(floor(m/k))
    N = if N_construct == "one"
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
                @show m, n, N, seed, decay, start
                try
                    if mode == "Boscia"
                        ODWB.solve_opt(
                            seed, 
                            m, 
                            n, 
                            time_limit, 
                            criterion, 
                            corr, 
                            integer_data=integer_data, 
                            N=N, 
                            smoothing_start=start,
                            smoothing_decay=decay,
                            smoothing_min=exp10(-200/m),
                            use_exclusion_criterion=option == "exclusion_criterion", 
                            use_heuristics=option == "all_heuristics", 
                            use_follow_subgradient_heu=option == "follow_subgradient", 
                            use_pipage_heu=option == "pipage_rounding", 
                            use_sr_rounding_heu=option == "sr_rounding", 
                            use_fedorov_heu=option == "fedorov", 
                            options_run=option != "baseline", 
                            mu_testing=option == "mu_testing")
                    elseif mode == "SCIP"
                        if criterion in ["A", "D", "E", "EF"]
                        error("SCIP OA does not work with the $(criterion)-optimal problems!")
                        end
                        ODWB.solve_opt_scip(seed, m, n, time_limit, criterion, corr, N=N)
                    elseif mode == "SCIPSDP"
                        if criterion in ["A", "D", "AF", "DF"]
                        error("SCIP SDP does not work with the $(criterion)-optimal problems!")
                        end
                        ODWB.solve_opt_scip_sdp(
                            seed, 
                            m, 
                            n, 
                            time_limit, 
                            criterion, 
                            corr, 
                            N=N, 
                            scip_sdp_mode=option == "oa" ? :oa : :bnb) #scip_sdp_mode=option == "oa" ? :oa : :bnb
                    elseif mode == "Pajarito"
                        ODWB.solve_opt_pajarito(seed, m, n, time_limit, criterion, corr, integer_data=integer_data, N=N)
                    elseif mode == "Custom"
                        if criterion in ["E", "EF"]
                            error("Co-BnB does not work with the $(criterion)-optimal problems!")
                        end
                        ODWB.solve_opt_custom(seed, m, n, time_limit, criterion, corr, N=N)
                    elseif mode == "SOCP"
                        if criterion in ["E", "EF"]
                            error("SOCP does not work with the $(criterion)-optimal problems!")
                        end
                        ODWB.solve_opt_socp(seed, m, n, time_limit, criterion, corr, N=N)
                    else 
                        error("Invalid mode!")
                    end
                catch e
                    println(e)
                    error_file = criterion * "_opt_" * mode * "_" * type * "_" * string(integer_data) * "_" * option * ".txt" 
                    open(error_file,"a") do io
                        println(io, seed, " ", m, " ", N, " ", mode, " : ", e)
                    end
                end
            end
        end
    end
end
