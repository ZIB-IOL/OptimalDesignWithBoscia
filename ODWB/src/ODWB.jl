module ODWB
using Boscia
using FrankWolfe
using Bonobo
using Random
using SCIP
using JuMP
using Hypatia
#using Clarabel
import Hypatia.Cones: vec_length, vec_copyto!, svec_length, svec_side
import Hypatia.Cones: smat_to_svec!, svec_to_smat!
const Cones = Hypatia.Cones
using Pajarito
using PajaritoExtras # https://github.com/chriscoey/PajaritoExtras.jl
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
using DoubleFloats
const MOI = MathOptInterface
const MOIU = MOI.Utilities
using StableRNGs

import MathOptSetDistances
const MOD = MathOptSetDistances

include("utilities.jl")
include("opt_design_boscia.jl")
include("scip_oa.jl")
include("opt_design_scip.jl")
include("opt_design_frank_wolfe.jl")
include("spectral_functions_JuMP.jl")
include("opt_design_pajarito.jl")
include("opt_design_custom_BB.jl")
include("opt_design_socp.jl")

end # module ODWB
