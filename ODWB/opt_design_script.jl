## Script for running sbatch

# ---------------------------------------------------------------------------
# E-Optimal — IND (+ optional CORR): scaled_mu A/B and reduced spectrum
# ---------------------------------------------------------------------------
num_experiments = [50, 80, 100, 120, 150]
criteria = ["E"]
data_types = ["IND"] #["IND", "CORR"]
solvers = ["Boscia"]
seeds = [0]
options = [
    "baseline",
    "reduced_spectrum_half",
    "reduced_spectrum_third",
    "scaled_mu",
    "reduced_spectrum_half_scaled_mu",
    "reduced_spectrum_third_scaled_mu",
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
# AGC — reduced spectrum + scaled_mu (L + A'DA spectrum for μ)
# ---------------------------------------------------------------------------
num_experiments = [80, 100, 150, 200]
criteria = ["AGC"]
data_types = ["IND", "CORR"]
solvers = ["Boscia"]
seeds = [0]
options = [
    "baseline",
    "reduced_spectrum_half",
    "reduced_spectrum_third",
    "scaled_mu",
    "reduced_spectrum_half_scaled_mu",
    "reduced_spectrum_third_scaled_mu",
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
# ACST / ACSTS — reduced spectrum + scaled_mu (L + A'DA spectrum for μ)
# ---------------------------------------------------------------------------
num_experiments = [10, 12, 15, 25, 40, 60, 100]
criteria = ["ACST", "ACSTS"]
data_types = ["IND"]
solvers = ["Boscia"]
seeds = [0]
options = [
    "baseline",
    "reduced_spectrum_half",
    "reduced_spectrum_third",
    "scaled_mu",
    "reduced_spectrum_half_scaled_mu",
    "reduced_spectrum_third_scaled_mu",
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
