using MKL
using ITensors, ITensorMPS
ITensors.BLAS.set_num_threads(1)
ITensors.Strided.set_num_threads(1)
if !isdefined(Main, :RandomUnitary)
    include("../src/RandomUnitary.jl")
    using .RandomUnitary
end

let 
    L, T = 10, 120
    ss = siteinds("S=1/2", L)
    psi = MPS(ComplexF64, ss, "Up")
    dent = NHDisentangler{Float64}(0.8, 0.2)
    obs = EntropyObserver{Float64}(L ÷ 2; n=1)

    @time timecorr, truncerr = mps_timecorrelation!(psi, T, 8L, dent, ("Z", 5, "Z", 5), obs; cutoff=1e-14, maxdim=10*L)
    @show timecorr
end