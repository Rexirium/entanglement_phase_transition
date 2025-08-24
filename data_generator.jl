using HDF5
include("entropy_calc.jl")

let 
    Ls = 8:8:48
    ps = 0.0:0.05:1.0
    ηs = 0.0:0.5:2.0
    nL, nprob, neta = length(Ls), length(ps), length(ηs)

    entropy_datas = zeros(nprob, nL, neta)
    for k in 1:neta
        η = ηs[k]
        for j in 1:nL
            l = Ls[j]
            tt, b = 4l, l ÷ 2
            mean_entropy = [entropy_mean(l, tt, p, η, b; numsamp=100) for p in ps]
            entropy_datas[:,j,k] .= mean_entropy
        end
    end

    h5open("entropy_data.h5", "w") do file
        write(file, "ps", collect(ps))
        write(file, "ηs", collect(ηs))
        write(file, "Ls", collect(Ls))
        write(file, "entropy_datas", entropy_datas)
    end
end
