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
ratio_para = [4,10]
time_limit = 3600 # one hour time limit

@show criterion, mode, corr

if !(criterion in ["A", "D", "DF", "AF"])
    error("Invalid criterion!")
end
for k in ratio_para
    n = Int(floor(m/k))
    for seed in 1:5
        @show m, n, seed
        try
            if mode == "Boscia"
                ODWB.solve_opt(seed, m, n, time_limit, criterion,corr)
            elseif mode == "SCIP"
                if criterion in ["A", "D"]
                   error("SCIP OA does not work with the optimal problems!")
                end
                ODWB.solve_opt_scip(seed, m, n, time_limit, criterion, corr)
            elseif mode == "Pajarito"
                ODWB.solve_opt_pajarito(seed, m, n, time_limit, criterion, corr)
            elseif mode == "Custom"
                ODWB.solve_opt_custom(seed, m, n, time_limit, criterion, corr)
            elseif mode == "SOCP"
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
