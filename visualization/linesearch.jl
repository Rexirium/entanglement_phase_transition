using MKL, LinearAlgebra
using HDF5, Interpolations
using CairoMakie

include("../linear_regress.jl")

let 
    L1, dL, L2 = 10, 2, 40
    nprob, neta = 21, 21

    file = h5open("data/nh_entrcorr_avg_L$(L1)_$(dL)_$(L2)_$(nprob)x$(neta).h5", "r")
    type_str = read(file, "datatype")
    ps = read(file, "params/ps")
    ηs = read(file, "params/ηs")
    Ls = read(file, "params/Ls")

    type = eval(Meta.parse(type_str))

    nL = length(Ls)
    entropy_datas = Array{type}(undef, nprob, neta, nL)
    entropy_error = Array{type}(undef, nprob, neta, nL)
    truncerrs = Array{type}(undef, nprob, neta, nL)

    for (i,l) in enumerate(Ls)
        @views entropy_datas[:, :, i] .= read(file, "L=$l/entr_means")
        @views entropy_error[:, :, i] .= read(file, "L=$l/entr_sems")
        @views truncerrs[:, :, i] .= read(file, "L=$l/truncerrs")
    end

    correlation_datas = read(file, "L=$L2/corr_means")
    correlation_error = read(file, "L=$L2/corr_sems")

    close(file)

    indices = zeros(type, nprob, neta)
end
