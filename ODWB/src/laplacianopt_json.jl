function _laplacianopt_catch_edge_error(num_nodes::Int, i::Int, j::Int, w_ij::Number)
    if (i > num_nodes) || (j > num_nodes)
        error("Node pair ($i,$j) does not match with total number of nodes, $num_nodes")
    end
    if !(isapprox(abs(w_ij), 0; atol=1e-6)) && (w_ij < 0)
        error("Graphs with negative weights are not supported")
    end
end

"""
    parse_laplacianopt_json(file_path::AbstractString) -> Dict{String,Any}

Parse a LaplacianOpt instance JSON file (same format as `LaplacianOpt.parse_file`).

Returned keys: `"num_nodes"`, `"adjacency_base_graph"`, `"adjacency_augment_graph"`.
"""
function parse_laplacianopt_json(file_path::AbstractString)
    data_dict = JSON.parsefile(file_path; dicttype=Dict{String,Any})

    haskey(data_dict, "num_nodes") ||
        error("Number of nodes is missing in the input data file")
    num_nodes = Int(data_dict["num_nodes"])

    adjacency_base_graph = zeros(Float64, num_nodes, num_nodes)
    adjacency_augment_graph = zeros(Float64, num_nodes, num_nodes)

    for edge in get(data_dict, "edges_existing", [])
        (i, j), w_ij = edge
        i, j = Int(i), Int(j)
        w_ij = isapprox(abs(w_ij), 0; atol=1e-6) ? 0.0 : Float64(w_ij)
        _laplacianopt_catch_edge_error(num_nodes, i, j, w_ij)
        adjacency_base_graph[i, j] = adjacency_base_graph[j, i] = w_ij
    end

    haskey(data_dict, "edges_to_augment") ||
        error("edges_to_augment is missing in the input data file")
    for edge in data_dict["edges_to_augment"]
        (i, j), w_ij = edge
        i, j = Int(i), Int(j)
        w_ij = isapprox(abs(w_ij), 0; atol=1e-6) ? 0.0 : Float64(w_ij)
        _laplacianopt_catch_edge_error(num_nodes, i, j, w_ij)
        adjacency_augment_graph[i, j] = adjacency_augment_graph[j, i] = w_ij
    end

    return Dict{String,Any}(
        "num_nodes" => num_nodes,
        "adjacency_base_graph" => adjacency_base_graph,
        "adjacency_augment_graph" => adjacency_augment_graph,
    )
end

function laplacianopt_instance_file(num_nodes::Integer, instance::Integer)
    # Do not pass `joinpath(pkgdir(...), ...)` as the default to `get(ENV, ...)`: the default is
    # evaluated eagerly, and `pkgdir(@__MODULE__)` is `nothing` when ODWB is loaded via `include`
    # from a script (not as a package), which breaks `joinpath`.
    root = if haskey(ENV, "LAPLACIANOPT_INSTANCES_ROOT") && !isempty(ENV["LAPLACIANOPT_INSTANCES_ROOT"])
        ENV["LAPLACIANOPT_INSTANCES_ROOT"]
    else
        p = pkgdir(@__MODULE__)
        p === nothing && error(
            "Set environment variable LAPLACIANOPT_INSTANCES_ROOT to the directory that contains " *
            "`{n}_nodes` subfolders (same layout as ODWB/data/laplacianopt_instances).",
        )
        joinpath(p, "data", "laplacianopt_instances")
    end
    joinpath(root, "$(num_nodes)_nodes", "$(num_nodes)_$(instance).json")
end
