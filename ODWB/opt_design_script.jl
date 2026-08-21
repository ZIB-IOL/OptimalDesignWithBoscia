## Script for running sbatch — production `optimized` presets
#
# OPTION=optimized picks criterion×type hyperparameters and writes CSVs as
#   boscia_optimized_<criterion>_optimality_...
# so baseline / A/B folders are never overwritten.
#
# Presets:
#   E CORR   — eigenvalue pruning + scaled_mu
#   E IND    — scaled_mu
#   AGC CORR — eigenvalue pruning
#   AGC IND  — reduced_spectrum_half + scaled_mu
#   ACST/ACSTS — rank_based_pruning + reduced_spectrum_half

# ---------------------------------------------------------------------------
# E-Optimal — IND + CORR
# ---------------------------------------------------------------------------
num_experiments = [50, 80, 100, 120, 150]
criteria = ["E"]
data_types = ["IND", "CORR"]
solvers = ["Boscia"]
seeds = [0]
options = ["optimized"]
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
# AGC — IND + CORR
# ---------------------------------------------------------------------------
num_experiments = [80, 100, 150, 200]
criteria = ["AGC"]
data_types = ["IND", "CORR"]
solvers = ["Boscia"]
seeds = [0]
options = ["optimized"]
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
options = ["optimized"]
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
