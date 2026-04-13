using MKL
using ITensors, ITensorMPS
ITensors.BLAS.set_num_threads(1)
ITensors.Strided.set_num_threads(1)
if !isdefined(Main, :RandomUnitary)
    include("../src/RandomUnitary.jl")
    using .RandomUnitary
end

function calculation_wrapper(lsize::Int, n::Real; nsamp::Int=100)
    corr_ops = ("Z", lsize ÷ 2, "Z", lsize ÷ 2)
    for i in 1:nsamp
        ss = siteinds("S=1/2", lsize)
        psi = MPS(ComplexF64, ss, "Up")
        mnt = PMMonitor{Float64}(lsize, n)
        tcorr, _ = timecorrelation!(psi, 4lsize, 2lsize, mnt, corr_ops)

    end
end

let 
    L, T = 16, 64
    ss = siteinds("S=1/2", L)
    psi = MPS(ComplexF64, ss, "Up")
    mnt = PMMonitor{Float64}(L, 20)
    obs = EntropyObserver{Float64}(L ÷ 2; n=1)

    @time timecorr, truncerr = timecorrelation!(psi, T, 2L, mnt, ("Z", 5, "Z", 5); cutoff=1e-14, maxdim=10*L)
    @show timecorr
end