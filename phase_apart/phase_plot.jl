using MKL
using Statistics
using Plots, LaTeXStrings
using HDF5

let 
    L1, dL, L2 = 6, 2, 18
    nprob, neta = 21, 21

    file = h5open("data/entropy_data_L$(L1)_$(dL)_$(L2)_$(nprob)x$(neta).h5", "r")
    type_str = read(file, "datatype")
    ps = read(file, "params/ps")
    ηs = read(file, "params/ηs")
    Ls = read(file, "params/Ls")

    type = eval(Meta.parse(type_str))

    nL = length(Ls)
    entropy_datas = Array{type}(undef, nprob, neta, nL)
    entropy_error = Array{type}(undef, nprob, neta, nL)
    for (i,l) in enumerate(Ls)
        entropy_datas[:, :, i] .= read(file, "L=$l/means")
        entropy_error[:, :, i] .= read(file, "L=$l/stds")
    end
    close(file)

    indices = Matrix{type}(undef, nprob, neta)

    for i in 1:nprob
        for j in 1:neta
            xs = collect(Ls)
            ys = entropy_datas[i, j, :]
            
        end
    end
end