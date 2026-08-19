## Script for running sbatch

### For the solver comparison

# Instance settings
# A- and D-Optimal Designs
#num_experiments = [50,60,80,100,120]
#criteria = ["D", "A", "DF", "AF"]
#data_types = ["CORR", "IND"]
#solvers = ["Boscia", "Pajarito", "SCIP", "Custom", "SOCP"]

# E-Opimal Design

# ---------------------------------------------------------------------------
# E-Optimal Design — CORR (incomplete reduced spectrum + all scaled options)
# ---------------------------------------------------------------------------
num_experiments = [50, 80, 100, 120, 150]
criteria = ["E"]
data_types = ["CORR"]
solvers = ["Boscia"]
seeds = [0]
options = [
    "reduced_spectrum_half",
    "reduced_spectrum_third",
    "reduced_spectrum_half_scaled",
    "reduced_spectrum_third_scaled",
    "scaled_mu",
    "baseline",
]
N_construct = ["one", "log"]

for criterion in criteria
    for data in data_types
        for solver in solvers
            for m in num_experiments
                for option in options
                    for seed in seeds
                        for N in N_construct
                            if (option == "exclusion_criterion" && solver == "SCIPSDP") || (option == "oa" && solver == "Boscia")
                                continue
                            end
                            run(`sbatch -A optimi -J B-E experiment.sbatch $criterion $solver $data $m $seed $option $N`)
                        end
                    end
                end
            end
        end
    end
end

# ---------------------------------------------------------------------------
# AGC — all options missing
# ---------------------------------------------------------------------------
num_experiments = [80, 100, 150, 200]
criteria = ["AGC"]
data_types = ["IND", "CORR"]
solvers = ["Boscia"]
seeds = [0]
options = [
    "reduced_spectrum_half",
    "reduced_spectrum_third",
    "reduced_spectrum_half_scaled",
    "reduced_spectrum_third_scaled",
    "scaled_input",
    "depth_first_search",
]
N_construct = ["one"]

for criterion in criteria
    for data in data_types
        for solver in solvers
            for m in num_experiments
                for option in options
                    for seed in seeds
                        for N in N_construct
                            if (option == "exclusion_criterion" && solver == "SCIPSDP") || (option == "oa" && solver == "Boscia")
                                continue
                            end
                            run(`sbatch -A optimi -J B-AGC experiment.sbatch $criterion $solver $data $m $seed $option $N`)
                        end
                    end
                end
            end
        end
    end
end

# ---------------------------------------------------------------------------
# ACST / ACSTS — all options missing
# ---------------------------------------------------------------------------
num_experiments = [10, 12, 15, 25, 40, 60, 100]
criteria = ["ACST", "ACSTS"]
data_types = ["IND"]
solvers = ["Boscia"]
seeds = [0]
options = [
    "reduced_spectrum_half",
    "reduced_spectrum_third",
    "reduced_spectrum_half_scaled",
    "reduced_spectrum_third_scaled",
    "scaled_input",
    "depth_first_search",
]
N_construct = ["one"]

for criterion in criteria
    for data in data_types
        for solver in solvers
            for m in num_experiments
                for option in options
                    for seed in seeds
                        for N in N_construct
                            if (option == "exclusion_criterion" && solver == "SCIPSDP") || (option == "oa" && solver == "Boscia")
                                continue
                            end
                            run(`sbatch -A optimi -J B-ACST experiment.sbatch $criterion $solver $data $m $seed $option $N`)
                        end
                    end
                end
            end
        end
    end
end
