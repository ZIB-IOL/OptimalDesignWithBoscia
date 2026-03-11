## Script for running sbatch

### For the solver comparison

# Instance settings
# A- and D-Optimal Designs
#num_experiments = [50,60,80,100,120]
#criteria = ["D", "A", "DF", "AF"]
#data_types = ["CORR", "IND"]
#solvers = ["Boscia", "Pajarito", "SCIP", "Custom", "SOCP"]

# E-Opimal Design

num_experiments = [50,80,100,120,150,170,200]#[50,80,100,120,150,170,200]  # for AGC [80, 100, 150, 200]
criteria = ["E"] # "EF"
data_types = ["IND", "CORR"]
solvers = ["Boscia"] #, "Pajarito" , "SCIPSDP"
seeds = [0] #[1,2,3,4,5]
options = ["exclusion_criterion"] #["baseline", "use_exclusion_criterion", "oa", "mu_testing"]
N_construct = ["one", "log", "rank_deficient"] #["one", "log", "rank_deficient"] 

# Connectivity 
#=
num_experiments = [80, 100, 150, 200]
criteria = ["AGC"]
data_types = ["IND"]
solvers = ["Boscia"]
seeds = [1,2,3,4,5]
options = ["exclusion_criterion"]
N_construct = ["one"] 
=#
# create instances
for criterion in criteria
    for data in data_types
        for solver in solvers
            for m in num_experiments
                for option in options
                    for seed in seeds
                        for N in N_construct
                            # 
                            if (option == "exclusion_criterion" && solver == "SCIPSDP") || (option == "oa" && solver == "Boscia")
                                continue
                            end
                            run(`sbatch -A optimi -J B-E experiment.sbatch $criterion $solver $data $m $seed $option $N`) # CB
                        end
                    end
                end
            end
        end
    end
end
