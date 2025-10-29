using MKL
using HDF5
MKL.set_num_threads(1)
include("entropy_calc.jl")

let
    # Parameters
    N = length(ARGS) == 0 ? 100 : parse(Int, ARGS[1])
    type = Float64

    p0::type, η0::type = 0.5, 0.5
    ps = collect(type, 0.0:0.05:1.0)
    ηs = collect(type, 0.0:0.05:1.0)
    Ls = collect(8:2:18)
    nprob, neta = length(ps), length(ηs)

    h5open("data/entropy_scale_L8_2_18.h5", "w") do file
        write(file, "datatype", string(type))
        grp = create_group(file, "params")
        write(grp, "N", N)  
        write(grp, "p0", p0)  
        write(grp, "η0", η0) 
        write(grp, "ps", ps)  
        write(grp, "ηs", ηs)    
        write(grp, "Ls", Ls)
    end

    for L in Ls
        cutoff = 1e-12 * L^3
        T = 4L
        # Store results
        prob_scales_mean = Vector{type}(undef, nprob)
        prob_scales_std = Vector{type}(undef, nprob)
        eta_scales_mean = Vector{type}(undef, neta)
        eta_scales_std = Vector{type}(undef, neta)

        # Calculate probability scaling
        for i in 1:nprob
            prob_scales_mean[i], prob_scales_std[i] = 
                entropy_mean_multi(L, T, ps[i], η0; numsamp=N, 
                    cutoff=cutoff, ent_cutoff=cutoff,  retstd=true, restype=type)
            println("L=$L, p=$(round(ps[i],digits=2)), η=0.5 done with $N samples.")
        end

        # Calculate eta scaling
        for i in 1:neta
            eta_scales_mean[i], eta_scales_std[i] = 
                entropy_mean_multi(L, T, p0, ηs[i]; numsamp=N, 
                    cutoff=cutoff, ent_cutoff=cutoff, retstd=true, restype=type)
            println("L=$L, p=0.50, η=$(round(ηs[i],digits=2)) done with $N samples.")
        end

        # Save data to HDF5 file
        h5open("data/entropy_scale_L8_2_18.h5", "cw") do file
            # create group if not exists
            grp = create_group(file, "results_L=$L")     

            write(grp, "prob_scales_mean", prob_scales_mean)
            write(grp, "prob_scales_std", prob_scales_std)
            write(grp, "eta_scales_mean", eta_scales_mean)
            write(grp, "eta_scales_std", eta_scales_std)
        end
    end
end
