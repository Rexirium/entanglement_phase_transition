include("../src/simulation.jl")

let 
    L, T = 10, 40
    b = L ÷ 2
    p, η = 0.9, 0.1
    N = 100

    @timev entropy_mean_multi(L, T, p, η; numsamp=N, retstd=true, restype=Float64)
end