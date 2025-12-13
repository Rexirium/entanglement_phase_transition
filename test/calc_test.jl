using MKL
MKL.set_num_threads(1)
include("../src/simulation.jl")

let 
    L, T = 10, 200
    b = L ÷ 2
    p, η = 0.5, 0.5
    N = length(ARGS) == 0 ? 100 : parse(Int, ARGS[1])

    res = @timev calculation_once(L, T, p, η; cutoff=1e-14,  retstd=true, restype=Float64)
end