using HDF5
using Base.Threads
include("entropy_calc.jl")

Threads.nthreads() = 50

let
    # Parameters
    Ls = 6:4:14
    p0, η0 = 0.5, 0.5
    ps = 0.0:0.05:1.0
    ηs = 0.0:0.1:2.0
    nprob, neta, nL = length(ps), length(ηs), length(Ls)
    
    # Store results
    prob_scales_mean = zeros(nprob, nL)
    prob_scales_std = zeros(nprob, nL)
    eta_scales_mean = zeros(neta, nL)
    eta_scales_std = zeros(neta, nL)

    # Create tasks array for all parameter combinations
    tasks_prob = [(i,j) for i in 1:nprob, j in 1:nL]
    tasks_eta = [(i,j) for i in 1:neta, j in 1:nL]

    # Calculate probability scaling
    for task in vec(tasks_prob)
        i, j = task
        l = Ls[j]
        tt, b = 4l, l ÷ 2
        p = ps[i]
        prob_scales_mean[i,j], prob_scales_std[i,j] = 
            entropy_mean_multi(l, tt, p, η0, b; numsamp=100, retstd=true)
        println("L=$l, p=$(round(p,digits=2)), η=0.5 done")
    end

    # Calculate eta scaling
    for task in vec(tasks_eta)
        i, j = task
        l = Ls[j]
        tt, b = 4l, l ÷ 2
        η = ηs[i]
        eta_scales_mean[i,j], eta_scales_std[i,j] = 
            entropy_mean_multi(l, tt, p0, η, b; numsamp=100, retstd=true)
        println("L=$l, p=0.50, η=$(round(η,digits=2)) done")
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
