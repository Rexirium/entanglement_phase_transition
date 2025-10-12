include("entropy_calc.jl")
#if abspath(PROGRAM_FILE) == @__FILE__
let
    # example usage
    L = 12
    T, b = 4L, L ÷ 2
    prob = 0.5
    eta = 0.5
    numsamp = 10

    @timev  entropy_mean(L, T, prob, eta; numsamp=numsamp, retstd=true)
    @timev  entropy_mean_multi(L, T, prob, eta; numsamp=numsamp, retstd=true)
end