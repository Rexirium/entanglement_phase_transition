using MKL
using HDF5
include("entropy_calc.jl")

let 
    # Model parameters
    Ls = 8:4:20
    ps = 0.0:0.05:1.0
    ηs = 0.0:0.2:1.0
    nL, nprob, neta = length(Ls), length(ps), length(ηs)
    
    # Store entanglement entropy data
    entropy_datas = zeros(nprob, nL, neta)
    
    # Create tasks array for all parameter combinations
    tasks = [(i,j,k) for i in 1:nprob, j in 1:nL, k in 1:neta]
    
    # Parallel computation over all parameter combinations
    for task in vec(tasks)
        i, j, k = task
        l = Ls[j]
        tt, b = 4l, l ÷ 2
        entropy_datas[i,j,k] = entropy_mean_multi(l, tt, ps[i], ηs[k], b; numsamp=100)
        
        # Progress tracking
        println("Completed: L=$(l), p=$(round(p,digits=2)), η=$(round(η,digits=2))")
    end

    h5open("entropy_data.h5", "w") do file
        write(file, "ps", collect(ps))
        write(file, "ηs", collect(ηs))
        write(file, "Ls", collect(Ls))
        write(file, "entropy_datas", entropy_datas)
    end
end
