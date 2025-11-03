using MKL
using HDF5
MKL.set_num_threads(1)
include("data_collapse.jl")

let 
    L1, dL, L2 = 6, 2, 18
    file = h5open("data/entropy_data_L$(L1)_$(dL)_$(L2)_11x11.h5", "r")
    type_str = read(file, "datatype")
    ps = read(file, "params/ps")
    ηs = read(file, "params/ηs")
    Ls = read(file, "params/Ls")

    type = eval(Meta.parse(type_str))

    nprob, neta, nL = length(ps), length(ηs), length(Ls)

    entropy_datas = Array{type}(undef, nprob, nL, neta)
    for (i, l) in enumerate(Ls)
        entropy_datas[:,:,i] .= read(file, "L=$l/means")
    end
    close(file)

    critical_params = data_collapse(entropy_datas, Ls, ps, ηs; numsamp=100)
    
    h5open("data/critical_params.h5", "w") do file
        write(file, "ηs", collect(ηs))
        write(file, "critical_params", critical_params)
    end
end