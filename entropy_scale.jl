using MKL
using HDF5
MKL.set_num_threads(1)
include("entropy_calc.jl")

let
    # Parameters
    L = ARGS[1] == nothing ? 8 : parse(Int, ARGS[1])
    T = ARGS[2] == nothing ? 4L : parse(Int, ARGS[2])
    N = ARGS[3] == nothing ? 100 : parse(Int, ARGS[3])

    p0, η0 = 0.5, 0.5
    ps = 0.0:0.05:1.0
    ηs = 0.0:0.05:1.0
    nprob, neta = length(ps), length(ηs)
    cutoff = 1e-12 * L^3
    
    # Store results
    prob_scales_mean = zeros(nprob)
    prob_scales_std = zeros(nprob)
    eta_scales_mean = zeros(neta)
    eta_scales_std = zeros(neta)

    # Calculate probability scaling
    for i in 1:nprob
        prob_scales_mean[i], prob_scales_std[i] = 
            entropy_mean_multi(L, T, ps[i], η0, L÷2; 
		    numsamp=N, cutoff=cutoff, ent_cutoff=cutoff,  retstd=true)
        println("L=$L, p=$(round(ps[i],digits=2)), η=0.5 done with $N samples.")
    end

    # Calculate eta scaling
    for i in 1:neta
        eta_scales_mean[i], eta_scales_std[i] = 
            entropy_mean_multi(L, T, p0, ηs[i], L÷2; 
		    numsamp=N, cutoff=cutoff, ent_cutoff=cutoff, retstd=true)
        println("L=$L, p=0.50, η=$(round(ηs[i],digits=2)) done with $N samples.")
    end

    # Save data to HDF5 file
    h5open("entropy_scale_test.h5", "cw") do file
        write(file, "ps", collect(ps))
        write(file, "ηs", collect(ηs))
        
        write(file, "prob_scales_mean", prob_scales_mean)
        write(file, "prob_scales_std", prob_scales_std)
        write(file, "eta_scales_mean", eta_scales_mean)
        write(file, "eta_scales_std", eta_scales_std)
    end
end
