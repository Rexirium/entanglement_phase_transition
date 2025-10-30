using HDF5
include("data_collapse.jl")

let 
    file = h5open("data/entropy_data.h5", "r")
    ps = read(file, "ps")
    ηs = read(file, "ηs")
    Ls = read(file, "Ls")
    entropy_datas = read(file, "entropy_datas")
    close(file)

    critical_params = data_collapse(entropy_datas, Ls, ps, ηs; numsamp=100)
    
    h5open("data/critical_params.h5", "w") do file
        write(file, "ηs", collect(ηs))
        write(file, "critical_params", critical_params)
    end
end