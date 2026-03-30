# Instance JSON files: copy `examples/instances/` from the LaplacianOpt repo into
# `ODWB/data/laplacianopt_instances/`, or set env var `LAPLACIANOPT_INSTANCES_ROOT` to that folder.

# possible values for number of nodes
# 5 8 9 10 12 15 25 40 60 100
function data_I(num_nodes::Int, instance::Int)
    file_path = laplacianopt_instance_file(num_nodes, instance)
    data_dict = parse_laplacianopt_json(file_path)
    augment_budget = (num_nodes - 1) # spanning tree constraint
    return data_dict, augment_budget
end

function data_ACST(num_nodes::Int, instance::Int; use_base_graph=false)
    file_path = laplacianopt_instance_file(num_nodes, instance)
    data_dict = parse_laplacianopt_json(file_path)
    W = data_dict["adjacency_augment_graph"]
    if num_nodes > 10
        W /= mean(abs.(W))
    else
        W /= maximum(abs.(W))
    end
    L = if use_base_graph data_dict["adjacency_base_graph"] else zeros(Float64, num_nodes, num_nodes) end
    num_present_edges = sum(Diagonal(L)) / 2
    potential_edges = Tuple{Int,Int}[]
    for i in 1:num_nodes
        for j in i:num_nodes
            if W[i,j] > 0
                push!(potential_edges, (i, j))
            end
        end
    end
    A = potential_edges_incidence_matrix(num_nodes, potential_edges; weights=W)
    @show size(A), size(L), num_present_edges
    return A, L, num_present_edges, potential_edges
end

"""
Builds the model maximizing the algebraic connectivity of a graph.
If `base_graph` is true, then consider that some indices are already set to one and augment the budget accordingly
"""
function algebraic_connectivity_model(seed, m, n; build_spanning_tree::Bool=true, use_base_graph=false, augment_budget::Int=-1)
    data_dict, _ = data_I(n, seed)
    n = data_dict["num_nodes"]
    W = data_dict["adjacency_augment_graph"]
    if build_spanning_tree
        augment_budget = n-1
    else
        if augment_budget < 0
            error("Provide the augment_budget keyword to build a general graph")
        end
    end
    @show augment_budget
    # we add the weight of the existing graph to the one to augment since the x_ij
    # of existing edges will be set to 1
    if use_base_graph
        adjacency_base_graph = data_dict["adjacency_base_graph"]
        W += adjacency_base_graph
    end
    @assert issymmetric(W)
    
    model = Model()
    @variable(model, gamma)
    @variable(model, x[i=1:n,j=1:n], Bin)
    @constraint(model, x .== x')
    # W in the article, renamed here to avoid confusion with the weights
    @variable(model, Y[1:n,1:n] in PSDCone())
    @constraint(model, [i=1:n],
        Y[i,i] == dot(W[i,:], x[i,:]) - gamma * (n-1) / n
    )
    #@constraint(model, Y[n,n] == - gamma * (n-1) / n)
    @constraint(model, [i=1:(n-1),j=(i+1):n],
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
        @constraint(model, sum(x[i,j] for i in 1:(n-1) for j in (i+1):n) == augment_budget + num_forced_edges)
    else
        @constraint(model, sum(x[i,j] for i in 1:(n-1) for j in (i+1):n) == augment_budget)
    end
    @objective(model, Max, gamma)
    return model, gamma, x
end

