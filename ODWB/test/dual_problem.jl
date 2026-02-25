using ODWB 
using JuMP
using Dualization
using Hypatia
using Test
using Mosek

seed = rand(UInt64)
@show seed
m = 20
n = Int(floor(sqrt(m)))
corr = false
N = 10

@show m, n

A, _, _, ub, _ = ODWB.build_data(seed, m, n, false, corr, N=N)


### Primal model ###
# SDP solver
opt = optimizer_with_attributes(Mosek.Optimizer, 
    MOI.Silent() => true, #!verbose,
)

primal_model = Model(opt)
# add variables
JuMP.@variable(primal_model, x[1:m])
JuMP.@variable(primal_model, t)
# we want to do s experiments
JuMP.@constraint(primal_model, sum(x) == 1)
@objective(primal_model, Max, t)
JuMP.@constraint(primal_model, x in MOI.Nonnegatives(m))

# PSD constraint: A' * diag(x) * A + t*I ⪰ 0
# This is equivalent to: A' * diag(x) * A - (-t)*I ⪰ 0
# We want to maximize t, so we minimize -t (the largest eigenvalue)
info_matrix = [
    JuMP.@expression(primal_model, 
        (i == j ? -t : 0.0) + sum(A[k, i] * x[k] * A[k, j] for k in 1:m)
    ) for i in 1:n, j in 1:n
]
# Add PSD constraint
JuMP.@constraint(primal_model, info_matrix in JuMP.PSDCone())

# solve 
optimize!(primal_model)
# query solution
status = termination_status(primal_model)
p_solution = objective_value(primal_model)
y = value.(x)

println("\n\nSolution of primal model")
@show status
@show p_solution
@show y


### Automatic Dual model ###
opt = optimizer_with_attributes(Mosek.Optimizer, 
    MOI.Silent() => true, #!verbose,
)

dual_model = Model(dual_optimizer(opt))
# add variables
JuMP.@variable(dual_model, x[1:m])
JuMP.@variable(dual_model, t)
# we want to do s experiments
JuMP.@constraint(dual_model, sum(x) == 1)
@objective(dual_model, Max, t)
JuMP.@constraint(dual_model, x in MOI.Nonnegatives(m))

# PSD constraint: A' * diag(x) * A + t*I ⪰ 0
# This is equivalent to: A' * diag(x) * A - (-t)*I ⪰ 0
# We want to maximize t, so we minimize -t (the largest eigenvalue)
info_matrix = [
    JuMP.@expression(dual_model, 
        (i == j ? -t : 0.0) + sum(A[k, i] * x[k] * A[k, j] for k in 1:m)
    ) for i in 1:n, j in 1:n
]
# Add PSD constraint
JuMP.@constraint(dual_model, info_matrix in JuMP.PSDCone())

# solve 
optimize!(dual_model)
# query solution
status = termination_status(dual_model)
a_d_solution = objective_value(dual_model)
y = value.(x)

println("\n\nSolution of automatic dual model")
@show status
@show a_d_solution
@show y


### Manual dual model (D-SDP) ###
# min_{λ, Z} λ
# s.t. Tr(Z) = 1
#      λ ≥ ⟨Z, v_i v_i^T⟩ for all i ∈ {1, ..., m}
#      Z ⪰ 0
opt = optimizer_with_attributes(Mosek.Optimizer, 
    MOI.Silent() => true, #!verbose,
)

dual_sdp_model = Model(opt)

# Variables
# λ: scalar variable
JuMP.@variable(dual_sdp_model, λ)
# Z: n×n positive semidefinite matrix
JuMP.@variable(dual_sdp_model, Z[1:n, 1:n], PSD)

# Constraint: Tr(Z) = 1
# The trace is the sum of diagonal elements
JuMP.@constraint(dual_sdp_model, sum(Z[i, i] for i in 1:n) == 1)

# Constraints: λ ≥ ⟨Z, v_i v_i^T⟩ for all i ∈ {1, ..., m}
# The inner product ⟨Z, v_i v_i^T⟩ = Tr(Z * (v_i * v_i^T)) = v_i^T * Z * v_i
# where v_i is the i-th row of A (i.e., A[i, :])
for i in 1:m
    v_i = A[i, :]  # i-th row of A
    # Compute v_i^T * Z * v_i as a quadratic expression
    inner_product = sum(v_i[j] * Z[j, k] * v_i[k] for j in 1:n for k in 1:n)
    JuMP.@constraint(dual_sdp_model, λ >= inner_product)
end

# Objective: minimize λ
@objective(dual_sdp_model, Min, λ)

# Solve
optimize!(dual_sdp_model)

# Query solution
status = termination_status(dual_sdp_model)
m_d_solution = objective_value(dual_sdp_model)
λ_val = value(λ)
Z_val = value.(Z)

println("\n\nSolution of manual dual SDP model (D-SDP)")
@show status
@show m_d_solution
@show λ_val
println("Z matrix:")
println(Z_val)


opt = optimizer_with_attributes(Mosek.Optimizer, 
    MOI.Silent() => true, #!verbose,
)

dual_sdp_s_model = Model(dual_optimizer(opt))

# Variables
# λ: scalar variable
JuMP.@variable(dual_sdp_s_model, λ)
# Z: n×n positive semidefinite matrix
JuMP.@variable(dual_sdp_s_model, Z[1:n, 1:n], PSD)

# Constraint: Tr(Z) = 1
# The trace is the sum of diagonal elements
JuMP.@constraint(dual_sdp_s_model, sum(Z[i, i] for i in 1:n) == 1)

# Constraints: λ ≥ ⟨Z, v_i v_i^T⟩ for all i ∈ {1, ..., m}
# The inner product ⟨Z, v_i v_i^T⟩ = Tr(Z * (v_i * v_i^T)) = v_i^T * Z * v_i
# where v_i is the i-th row of A (i.e., A[i, :])
for i in 1:m
    v_i = A[i, :]  # i-th row of A
    # Compute v_i^T * Z * v_i as a quadratic expression
    inner_product = sum(v_i[j] * Z[j, k] * v_i[k] for j in 1:n for k in 1:n)
    JuMP.@constraint(dual_sdp_s_model, λ >= inner_product)
end

# Objective: minimize λ
@objective(dual_sdp_s_model, Min, λ)

# Solve
optimize!(dual_sdp_s_model)

# Query solution
status = termination_status(dual_sdp_s_model)
m_d_s_solution = objective_value(dual_sdp_s_model)
λ_val = value(λ)
Z_val = value.(Z)

println("\n\nSolution of dual of manual dual SDP model (D-SDP)")
@show status
@show m_d_s_solution
@show λ_val
println("Z matrix:")
println(Z_val)


@test isapprox(p_solution, m_d_solution, atol=1e-6)
@test isapprox(p_solution, m_d_s_solution, atol=1e-6)
@test isapprox(a_d_solution, m_d_solution, atol=1e-6)
@test isapprox(a_d_solution, m_d_s_solution, atol=1e-6)

### Primal model ###
# SDP solver
opt = optimizer_with_attributes(Mosek.Optimizer, 
    MOI.Silent() => true, #!verbose,
)

primal_model = Model(opt)
# add variables
JuMP.@variable(primal_model, x[1:m])
JuMP.@variable(primal_model, t)
# we want to do s experiments
JuMP.@constraint(primal_model, sum(x) == N)
JuMP.@constraint(primal_model, x <= ub)
@objective(primal_model, Max, t)
JuMP.@constraint(primal_model, x in MOI.Nonnegatives(m))

# PSD constraint: A' * diag(x) * A + t*I ⪰ 0
# This is equivalent to: A' * diag(x) * A - (-t)*I ⪰ 0
# We want to maximize t, so we minimize -t (the largest eigenvalue)
info_matrix = [
    JuMP.@expression(primal_model, 
        (i == j ? -t : 0.0) + sum(A[k, i] * x[k] * A[k, j] for k in 1:m)
    ) for i in 1:n, j in 1:n
]
# Add PSD constraint
JuMP.@constraint(primal_model, info_matrix in JuMP.PSDCone())

# solve 
optimize!(primal_model)
# query solution
status = termination_status(primal_model)
p_solution = objective_value(primal_model)
y = value.(x)

println("\n\nSolution of primal model")
@show status
@show p_solution
@show y


### Automatic Dual model ###
opt = optimizer_with_attributes(Mosek.Optimizer, 
    MOI.Silent() => true, #!verbose,
)

dual_model = Model(dual_optimizer(opt))
# add variables
JuMP.@variable(dual_model, x[1:m])
JuMP.@variable(dual_model, t)
# we want to do s experiments
JuMP.@constraint(dual_model, sum(x) == N)
@objective(dual_model, Max, t)
JuMP.@constraint(dual_model, x in MOI.Nonnegatives(m))
JuMP.@constraint(dual_model, x <= ub)

# PSD constraint: A' * diag(x) * A + t*I ⪰ 0
# This is equivalent to: A' * diag(x) * A - (-t)*I ⪰ 0
# We want to maximize t, so we minimize -t (the largest eigenvalue)
info_matrix = [
    JuMP.@expression(dual_model, 
        (i == j ? -t : 0.0) + sum(A[k, i] * x[k] * A[k, j] for k in 1:m)
    ) for i in 1:n, j in 1:n
]
# Add PSD constraint
JuMP.@constraint(dual_model, info_matrix in JuMP.PSDCone())

# solve 
optimize!(dual_model)
# query solution
status = termination_status(dual_model)
a_d_solution = objective_value(dual_model)
y = value.(x)

println("\n\nSolution of automatic dual model")
@show status
@show a_d_solution
@show y


### Manual dual model (D-SDP) ###
# min_{λ, Z} λ
# s.t. Tr(Z) = 1
#      λ ≥ ⟨Z, v_i v_i^T⟩ for all i ∈ {1, ..., m}
#      Z ⪰ 0
opt = optimizer_with_attributes(Mosek.Optimizer, 
    MOI.Silent() => true, #!verbose,
)

dual_sdp_model = Model(opt)

# Variables
# λ: scalar variable
JuMP.@variable(dual_sdp_model, λ)
# Z: n×n positive semidefinite matrix
JuMP.@variable(dual_sdp_model, Z[1:n, 1:n], PSD)

# Constraint: Tr(Z) = 1
# The trace is the sum of diagonal elements
JuMP.@constraint(dual_sdp_model, sum(Z[i, i] for i in 1:n) == 1)

# Constraints: λ ≥ ⟨Z, v_i v_i^T⟩ for all i ∈ {1, ..., m}
# The inner product ⟨Z, v_i v_i^T⟩ = Tr(Z * (v_i * v_i^T)) = v_i^T * Z * v_i
# where v_i is the i-th row of A (i.e., A[i, :])
#for i in 1:m
#    v_i = A[i, :]  # i-th row of A
    # Compute v_i^T * Z * v_i as a quadratic expression
#    inner_product = sum(v_i[j] * Z[j, k] * v_i[k] for j in 1:n for k in 1:n)
#    JuMP.@constraint(dual_sdp_model, λ >= inner_product)
#end

# Objective: minimize λ
@objective(dual_sdp_model, Min, λ*N + sum(ub[i] * max(0, sum(A[i, :]' * Z * A[i, :]) - λ) for i in 1:m))

# Solve
optimize!(dual_sdp_model)

# Query solution
status = termination_status(dual_sdp_model)
m_d_solution = objective_value(dual_sdp_model)
λ_val = value(λ)
Z_val = value.(Z)

println("\n\nSolution of manual dual SDP model (D-SDP)")
@show status
@show m_d_solution
@show λ_val
println("Z matrix:")
println(Z_val)


opt = optimizer_with_attributes(Mosek.Optimizer, 
    MOI.Silent() => true, #!verbose,
)

dual_sdp_s_model = Model(dual_optimizer(opt))

# Variables
# λ: scalar variable
JuMP.@variable(dual_sdp_s_model, λ)
# Z: n×n positive semidefinite matrix
JuMP.@variable(dual_sdp_s_model, Z[1:n, 1:n], PSD)

# Constraint: Tr(Z) = 1
# The trace is the sum of diagonal elements
JuMP.@constraint(dual_sdp_s_model, sum(Z[i, i] for i in 1:n) == 1)

# Constraints: λ ≥ ⟨Z, v_i v_i^T⟩ for all i ∈ {1, ..., m}
# The inner product ⟨Z, v_i v_i^T⟩ = Tr(Z * (v_i * v_i^T)) = v_i^T * Z * v_i
# where v_i is the i-th row of A (i.e., A[i, :])
#for i in 1:m
#    v_i = A[i, :]  # i-th row of A
    # Compute v_i^T * Z * v_i as a quadratic expression
#    inner_product = sum(v_i[j] * Z[j, k] * v_i[k] for j in 1:n for k in 1:n)
#    JuMP.@constraint(dual_sdp_s_model, λ >= inner_product)
#end

# Objective: minimize λ
@objective(dual_sdp_s_model, Min, λ*N + sum(ub[i] * max(0, sum(A[i, :]' * Z * A[i, :]) - λ) for i in 1:m))

# Solve
optimize!(dual_sdp_s_model)

# Query solution
status = termination_status(dual_sdp_s_model)
m_d_s_solution = objective_value(dual_sdp_s_model)
λ_val = value(λ)
Z_val = value.(Z)

println("\n\nSolution of dual of manual dual SDP model (D-SDP)")
@show status
@show m_d_s_solution
@show λ_val
println("Z matrix:")
println(Z_val)


@test isapprox(p_solution, m_d_solution, atol=1e-6)
@test isapprox(p_solution, m_d_s_solution, atol=1e-6)
@test isapprox(a_d_solution, m_d_solution, atol=1e-6)
@test isapprox(a_d_solution, m_d_s_solution, atol=1e-6)
