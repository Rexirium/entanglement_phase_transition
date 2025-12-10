using MKL
MKL.set_num_threads(1)
include("../src/simulation.jl")

let 
    L, T = 10, 40
    b = L ÷ 2
    p, η = 0.5, 0.5
    N = length(ARGS) == 0 ? 100 : parse(Int, ARGS[1])

    mean, std = @timev entropy_mean_multi(L, T, p, η; numsamp=N, cutoff=1e-14,  retstd=true, restype=Float64)
    println("Entropy: $mean ± $std, with $T timesteps and $N samples.")
end