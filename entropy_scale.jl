using HDF5
include("entropy_calc.jl")

let
    # Parameters
    Ls = 6:4:22
    p0, η0 = 0.5, 0.5
    ps = 0.0:0.05:1.0
    ηs = 0.0:0.1:2.0
    nprob, neta, nL = length(ps), length(ηs), length(Ls)
    # Store results
    prob_scales_mean = zeros(nprob, nL)
    prob_scales_std = zeros(nprob, nL)
    eta_scales_mean = zeros(neta, nL)
    eta_scales_std = zeros(neta, nL)
    # Calculate entanglement entropy (mean and std) for each parameter set
    for j in 1:nL
        l = Ls[j]
        tt, b = 4l, l ÷ 2

        for i in 1:nprob
            p = ps[i]
            prob_scales_mean[i,j], prob_scales_std[i,j] = 
                entropy_mean(l, tt, p, η0, b; numsamp=10, retstd=true)
            println("L=$l, p=$(round(p,digits=2)), η=0.5 done")
        end

        for i in 1:neta
            η = ηs[i]
            eta_scales_mean[i,j], eta_scales_std[i,j] = 
                entropy_mean(l, tt, p0, η, b; numsamp=10, retstd=true)
            println("L=$l, p=0.50, η=$(round(η,digits=2)) done")
        end
    end
    # Save data to HDF5 file
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