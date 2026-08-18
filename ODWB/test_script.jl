# Define instance
ENV["MODE"] = "Boscia"
ENV["CRITERION"] = "E"
ENV["TYPE"] = "CORR"
m = 50
ENV["DIMENSION"] = string(m)
ENV["SEED"] = "1"
ENV["OPTION"] = "reduced_spectrum_half"
ENV["N"] = "one"

include("run_optimal_design.jl")