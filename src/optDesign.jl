module optDesign

using Boscia
using FrankWolfe
using Bonobo
using Random
using SCIP
using JuMP
using Hypatia
import Hypatia.Cones: vec_length, vec_copyto!, svec_length, svec_side
import Hypatia.Cones: smat_to_svec!, svec_to_smat!
const Cones = Hypatia.Cones
using Pajarito
using PajaritoExtras
using HiGHS
using LinearAlgebra
using Statistics
using Distributions
import MathOptInterface
using Printf
using Dates
using Test
using DataFrames
using CSV
const MOI = MathOptInterface
const MOIU = MOI.Utilities

import MathOptSetDistances
const MOD = MathOptSetDistances

# for bug hunting
using Infiltrator

include("utilities.jl")
include("opt_design_boscia.jl")
include("scip_oa.jl")
include("opt_design_scip.jl")
include("opt_design_frank_wolfe.jl")
include("opt_design_hypatia.jl")
include("spectral_functions_JuMP.jl")
include("opt_design_pajarito.jl")
include("opt_design_custom_BB.jl")
include("opt_design_socp.jl")

end # module optDesign