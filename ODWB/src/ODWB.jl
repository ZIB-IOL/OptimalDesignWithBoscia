module ODWB
using Boscia
using FrankWolfe
using Bonobo
using CombinatorialLinearOracles
const CO = CombinatorialLinearOracles
using Random
using SCIP
using JuMP
using Hypatia
#using Clarabel
import Hypatia.Cones: vec_length, vec_copyto!, svec_length, svec_side
import Hypatia.Cones: smat_to_svec!, svec_to_smat!
const Cones = Hypatia.Cones
using Pajarito
#using PajaritoExtras # https://github.com/chriscoey/PajaritoExtras.jl
using HiGHS
using LinearAlgebra
using SparseArrays
using Statistics
using Distributions
import MathOptInterface
using Printf
using Dates
using Test
using DataFrames
using CSV
using DoubleFloats
using LogExpFunctions
const MOI = MathOptInterface
const MOIU = MOI.Utilities
using StableRNGs
using Dualization
using JSON
using Graphs
using Arpack
using Suppressor

import MathOptSetDistances
const MOD = MathOptSetDistances

# path local /Users/deborah/mosek/mosek/11.0/tools/platform/osxaarch64/bin
# path cluster /software/mosek/10.2/tools/platform/linux64x86
using Mosek 
using MosekTools

include("laplacianopt_json.jl")
include("utilities.jl")
include("algebraic_connectivity_tree.jl")
include("heuristics.jl")
include("exclusion_criterion.jl")
include("opt_design_boscia.jl")
include("scip_oa.jl")
include("opt_design_scip.jl")
include("opt_design_scip_sdp.jl")
include("opt_design_frank_wolfe.jl")
include("spectral_functions_JuMP.jl")
include("opt_design_pajarito.jl")
include("opt_design_custom_BB.jl")
include("opt_design_socp.jl")

end # module ODWB
