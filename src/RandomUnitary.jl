module RandomUnitary

using ITensors, ITensorMPS
using LinearAlgebra: diagm, Diagonal, I
using StaticArrays: SVector
using Random: shuffle
using SparseArrays: sparse, blockdiag

# ITensors.BLAS.set_num_threads(1)
# ITensors.Strided.set_num_threads(1)

include("entanglement.jl")
include("correlation.jl")
include("observers.jl")
include("time_evolution.jl")
include("simulation.jl")

export ent_entropy, negativity, concurrence, zeroth_entropy, mutual_information, correlation, correlation_vec
export AbstractObserver, EntropyObserver, EntropyAverager, EntrCorrObserver, EntrCorrAverager, mps_monitor!
export AbstractDisentangler, NHDisentangler, NHCNOTDisentangler, disentangle!, mps_evolve!, mps_timecorrelation!
export CalcResult, AbstractResult, EntropyResults, EntrCorrSample, mps_results!, calculation_mean, calculation_mean_multi

end # RandomUnitary

