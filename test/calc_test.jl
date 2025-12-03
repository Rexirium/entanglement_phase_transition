include("../src/simulation.jl")

let 
    L = 10
    T, b = 4L, L ÷ 2
    p, η = 0.5, 0.5
    N = 100

    @timev calculation_mean(L, T, p, η; numsamp=N, retstd=true, restype=Float64)
    @timev calculation_mean_multi(L, T, p, η; numsamp=N, retstd=true, restype=Float64)
end