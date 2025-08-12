# Script for running the experiments
using ODWB
using Printf
using Test
using DataFrames
using CSV

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
ratio_para = criterion in ["E", "EF"] ? [1] : [4,10]
time_limit = 3600 # one hour time limit
seeds = seed == 0 ? collect(1:5) : [seed]

@show criterion, mode, corr

if !(criterion in ["A", "D", "DF", "AF", "E", "EF"])
    error("Invalid criterion!")
end
for k in ratio_para
    n = k == 1 ? Int(floor(sqrt(m))) : Int(floor(m/k))
    for seed in seeds
        @show m, n, seed
        try
            if mode == "Boscia"
                ODWB.solve_opt(seed, m, n, time_limit, criterion, corr, integer_data=integer_data)
            elseif mode == "SCIP"
                if criterion in ["A", "D", "E", "EF"]
                   error("SCIP OA does not work with the $(criterion)-optimal problems!")
                end
                ODWB.solve_opt_scip(seed, m, n, time_limit, criterion, corr)
            elseif mode == "Pajarito"
                ODWB.solve_opt_pajarito(seed, m, n, time_limit, criterion, corr, integer_data=integer_data)
            elseif mode == "Custom"
                if criterion in ["E", "EF"]
                    error("Co-BnB does not work with the $(criterion)-optimal problems!")
                 end
                ODWB.solve_opt_custom(seed, m, n, time_limit, criterion, corr)
            elseif mode == "SOCP"
                if criterion in ["E", "EF"]
                    error("SOCP does not work with the $(criterion)-optimal problems!")
                 end
                ODWB.solve_opt_socp(seed, m, n, time_limit, criterion, corr)
            else 
                error("Invalid mode!")
            end
        catch e
            println(e)
            error_file = criterion * "_opt_" * mode * "_" * type * ".txt" 
            open(error_file,"a") do io
                println(io, seed, " ", m, " ", mode, " : ", e)
            end
        end
    end
end
