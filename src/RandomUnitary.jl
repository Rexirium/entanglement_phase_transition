module RandomUnitary

using ITensors, ITensorMPS
using ITensors: svd
using LinearAlgebra: diagm
using Random: shuffle
using SparseArrays: sparse, blockdiag

# ITensors.BLAS.set_num_threads(1)
# ITensors.Strided.set_num_threads(1)

include("entanglement.jl")
include("correlation.jl")
include("operators.jl")
include("monitors.jl")
include("observers.jl")
include("time_evolution.jl")
include("time_correlation.jl")
# include("simulation.jl")

export ent_entropy, negativity, concurrence, zeroth_entropy, mutual_information, correlation, correlation_vec
export proj_measure!, weak_measure!, AbstractMonitor, NHMonitor, PMMonitor, monitor!
export AbstractObserver, EntropyObserver, EntropyAverager, EntrCorrObserver, EntrCorrAverager, mps_record!
export timeevolve!, timecorrelation!
# export CalcResult, AbstractResult, EntropyResults, mps_results!, calculation_mean, calculation_mean_multi

end # RandomUnitary
