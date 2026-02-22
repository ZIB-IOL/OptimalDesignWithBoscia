## Script for running sbatch

### For the solver comparison

# Instance settings
# A- and D-Optimal Designs
#num_experiments = [50,60,80,100,120]
#criteria = ["D", "A", "DF", "AF"]
#data_types = ["CORR", "IND"]
#solvers = ["Boscia", "Pajarito", "SCIP", "Custom", "SOCP"]

# E-Opimal Design
num_experiments = [50,80,100,120,150]
criteria = ["E"] # "EF"
data_types = ["IND", "CORR"]
integer_data = [false]  #[0, 1]
solvers = ["Boscia", "SCIPSDP"] #, "Pajarito"
seeds = [0] #[1,2,3,4,5]
options = ["baseline", "use_exclusion_criterion", "oa"]
N_construct = ["one", "log", "rank_deficient"] 


# create instances
for criterion in criteria
    for data in data_types
        for solver in solvers
            for m in num_experiments
                for option in options
                    for seed in seeds
                        for int_data in integer_data
                            for N in N_construct
                                # 
                                if (options == "use_exclusion_criterion" && mode == "SCIPSDP") || (options == "oa" && mode == "Boscia")
                                    continue
                                end
                                run(`sbatch -A optimi -J E-Opt experiment.sbatch $criterion $solver $data $m $int_data $seed $option $N`) # CB
                            end
                        end
                    end
                end
            end
        end
    end
end
