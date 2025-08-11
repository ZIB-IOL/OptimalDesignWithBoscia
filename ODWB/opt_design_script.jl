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
criteria = ["E", "EF"]
data_types = ["IND"]
integer_data = [0, 1]
solvers = ["Boscia", "Pajarito"]
seeds = [0] #[1,2,3,4,5]


# create instances
for criterion in criteria
    for data in data_types
        for solver in solvers
            for m in num_experiments
                for seed in seeds
                    for int_data in integer_data
                        # 
                        run(`sbatch -A optimi -J Co-Fusion experiment.sbatch $criterion $solver $data $m $int_data $seed`) # CB
                    end
                end
            end
        end
    end
end
