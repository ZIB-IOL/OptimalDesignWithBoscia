import LaplacianOpt as LOpt
using JuMP
using LinearAlgebra

# possible values for number of nodes
# 5 8 9 10 12 15 25 40 60 100
function data_I(num_nodes::Int, instance::Int)
    # Data format has to be as given in this JSON file
    file_path =
    joinpath(dirname(pathof(LOpt)), "../examples/instances/$(num_nodes)_nodes/$(num_nodes)_$(instance).json")
    data_dict = LOpt.parse_file(file_path)
    augment_budget = (num_nodes - 1) # spanning tree constraint
    return data_dict, augment_budget
end

data_dict, augment_budget = data_I(8, 1)


"""
Builds the model maximizing the algebraic connectivity of a graph.
If `base_graph` is true, then consider that some indices are already set to one and augment the budget accordingly
"""
function algebraic_connectivity_model(data_dict, build_spanning_tree::Bool=true; use_base_graph=false, augment_budget::Int=-1)
    n = data_dict["num_nodes"]
    W = data_dict["adjacency_augment_graph"]
    if build_spanning_tree
        augment_budget = n-1
    else
        if augment_budget < 0
            error("Provide the augment_budget keyword to build a general graph")
        end
    end
    # we add the weight of the existing graph to the one to augment since the x_ij
    # of existing edges will be set to 1
    if use_base_graph
        adjacency_base_graph = data_dict["adjacency_base_graph"]
        W += adjacency_base_graph
    end
    @assert issymmetric(W)
    
    m = Model()
    @variable(m, gamma)
    @variable(m, x[i=1:n,j=1:n], Bin)
    @constraint(m, x .== x')
    # W in the article, renamed here to avoid confusion with the weights
    @variable(m, Y[1:n,1:n] in PSDCone())
    @constraint(m, [i=1:n],
        Y[i,i] == dot(W[i,:], x[i,:]) - gamma * (n-1) / n
    )
    @constraint(m, [i=1:n,j=1:n],
        Y[i,j] == -W[i,j]*x[i,j] + gamma / n,
    )
    # spanning tree constraint
    # can be changed later to more flexibility
    if use_base_graph
        num_forced_edges = 0
        for i in 1:(n-1)
            for j in (i+1):n
                if adjacency_base_graph[i,j] > 0
                    JuMP.fix(x[i,j], 1; force=true)
                    JuMP.fix(x[j,i], 1; force=true)
                    num_forced_edges += 1
                end
            end
        end
        @constraint(m, sum(x) == augment_budget + num_forced_edges)
    else
        @constraint(m, sum(x) == augment_budget)
    end
    @objective(m, Max, gamma)
    return m
end

