include("../src/entropy_calc.jl")

let 
    L = 12
    T, b = 4L, L ÷ 2
    p, η = 0.5, 0.5
    N = 100

    @timev entropy_mean(L, T, p, η; numsamp=N, retstd=true, restype=Float64)
    @timev entropy_mean_multi(L, T, p, η; numsamp=N, retstd=true, restype=Float64)
end