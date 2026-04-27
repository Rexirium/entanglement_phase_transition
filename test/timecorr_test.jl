using MKL
using ITensors, ITensorMPS
using Statistics
using CairoMakie

ITensors.BLAS.set_num_threads(1)
ITensors.Strided.set_num_threads(1)
if !isdefined(Main, :RandomUnitary)
    include("../src/RandomUnitary.jl")
    using .RandomUnitary
end

mutable struct MyObserver{T} <: AbstractObserver
    b::Int
    n::Real
    op::String
    entropies::Vector{T}
    expvals::Vector{T}
    accept::Bool

    MyObserver{T}(b::Int, n::Real, op::String) where T<:Real = new{T}(b, n, op, T[], T[], true)
end

function RandomUnitary.mps_record!(obs::MyObserver, psi::MPS, t::Int, truncerr::Real)
    push!(obs.entropies, ent_entropy(psi, obs.b))
    push!(obs.expvals, expected(psi, obs.op, obs.b))
end

function calculation_wrapper(lsize::Int, n::Real; nsamp::Int=100)
    corr_ops = ("Z", lsize ÷ 2, "Z", lsize ÷ 2)
    ttotal = 8lsize

    entropies = Matrix{Float64}(undef, ttotal + 1, nsamp)
    expectvals = Matrix{Float64}(undef, ttotal + 1, nsamp)
    timecorrs = Matrix{Float64}(undef, lsize + 1, nsamp)
    spatcorrs = Matrix{Float64}(undef, lsize, nsamp)

    Threads.@threads for i in 1:nsamp
        ss = siteinds("S=1/2", lsize)
        psi = MPS(ComplexF64, ss, "Up")
        mnt = PMMonitor{Float64}(lsize, n)
        obs = MyObserver{Float64}(lsize ÷ 2, n, "Z")
        tcorr, _ = timecorrelation!(psi, ttotal, ttotal - lsize, mnt, corr_ops, obs; maxdim = lsize * lsize)

        entropies[:, i] = obs.entropies
        expectvals[:, i] = obs.expvals
        timecorrs[:, i] = abs2.(tcorr)
        spatcorrs[:, i] = abs2.(correlation_site(psi, "Z", "Z"))
    end
    entropy_mean = mean(entropies, dims=2)[:, 1]
    entropy_sems = stdm(entropies, entropy_mean; dims=2)[:, 1] / sqrt(nsamp)
    expectval_mean = mean(expectvals, dims=2)[:, 1]
    expectval_sems = stdm(expectvals, expectval_mean; dims=2)[:, 1] / sqrt(nsamp)
    timecorr_mean = mean(timecorrs, dims=2)[:, 1]
    timecorr_sems = stdm(timecorrs, timecorr_mean; dims=2)[:, 1] / sqrt(nsamp)
    spatcorr_mean = mean(spatcorrs, dims=2)[:, 1]
    spatcorr_sems = stdm(spatcorrs, spatcorr_mean; dims=2)[:, 1] / sqrt(nsamp)

    return entropy_mean, entropy_sems, expectval_mean, expectval_sems, timecorr_mean, timecorr_sems, spatcorr_mean, spatcorr_sems
end

let 
    L = 12
    n = 7
    T = 8L
    @time res = calculation_wrapper(L, n)
    
    fig = Figure(size=(1000, 800))
    ax1 = Axis(fig[1, 1], xlabel=L"t", ylabel=L"S(L/2)", title="Entanglement Entropy, L = $L")
    lines!(ax1, 0 : T, res[1], linewidth=1.5, label="n = $n")
    errorbars!(ax1, 0 : T, res[1], res[2], linewidth=1.5)
    axislegend(ax1, position=:rt)

    ax2 = Axis(fig[1, 2], xlabel=L"t", ylabel=L"⟨Z_{L/2}⟩", 
        title="Expectation Value")
    lines!(ax2, 0 : T, res[3], linewidth=1.5, label="n = $n")
    errorbars!(ax2, 0 : T, res[3], res[4], linewidth=1.5)
    axislegend(ax2, position=:rt)

    ax3 = Axis(fig[2, 1], xlabel=L"t", ylabel=L"C(t=8L, t'=7L)", 
        title="Time Correlation", limits=(nothing, (0.0, 1.0)))
    lines!(ax3, 0 : L, res[5], linewidth=1.5, label="n = $n")
    errorbars!(ax3, 0 : L, res[5], res[6], linewidth=1.5)
    axislegend(ax3, position=:rt)

    ax4 = Axis(fig[2, 2], xlabel=L"r", ylabel=L"C(r)", 
        limits=(nothing, (0.0, 1.0)), title="Spatial Correlation")
    lines!(ax4, 0 : (L-1), res[7], linewidth=1.5, label="n = $n")
    errorbars!(ax4, 0 : (L-1), res[7], res[8], linewidth=1.5)
    axislegend(ax4, position=:rt)
    fig
    # save("figures/timecorr_results_$L.png", fig)
end