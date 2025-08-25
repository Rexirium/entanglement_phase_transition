using HDF5
include("entropy_calc.jl")

let
    Ls = 10:10:50
    p0, η0 = 0.5, 0.5
    ps = 0.0:0.05:1.0
    ηs = 0.0:0.5:2.0
    nprob, neta, nL = length(ps), length(ηs), length(Ls)

    prob_scales_mean = zeros(nprob, nL)
    prob_scales_std = zeros(nprob, nL)
    eta_scales_mean = zeros(neta, nL)
    eta_scales_std = zeros(neta, nL)

    for j in 1:nL
        l = Ls[j]
        tt, b = 4l, l ÷ 2

        for i in 1:nprob
            prob_scales_mean[i,j], prob_scales_std[i,j] = 
                entropy_mean(l, tt, ps[i], η0, b; numsamp=10, retstd=true)
        end

        for i in 1:neta
            eta_scales_mean[i,j], eta_scales_std[i,j] = 
                entropy_mean(l, tt, p0, ηs[i], b; numsamp=10, retstd=true)
        end
    end

    h5open("entropy_scale_data.h5", "w") do file
        write(file, "ps", collect(ps))
        write(file, "ηs", collect(ηs))
        write(file, "Ls", collect(Ls))
        
        write(file, "prob_scales_mean", prob_scales_mean)
        write(file, "prob_scales_std", prob_scales_std)
        write(file, "eta_scales_mean", eta_scales_mean)
        write(file, "eta_scales_std", eta_scales_std)
    end
end