using MKL
using ITensors, ITensorMPS
include("../src/entanglement.jl")
include("../src/correlation.jl")
using CairoMakie

mutable struct MyObserver <: AbstractObserver
    entropies::Vector{Float64}
    correlations::Vector{Float64}
    maxbonds::Vector{Int}

    MyObserver() = new(Float64[], Float64[], Int[])
end

function mps_record!(obs::MyObserver, psi::MPS, t::Int)
    b = length(psi) ÷ 2
    push!(obs.entropies, ent_entropy(psi, b))
    push!(obs.correlations, correlation(psi, "Z", "Z", b, b + 1))
    push!(obs.maxbonds, maxlinkdim(psi))
end

function make_initialstate(ss::Vector{<:Index}, period::Int, start::String)
    if start == "Up"
        return MPS(ss, n -> (mod1(n, period) == 1 ? "Up" : "Dn"))
    elseif start == "Dn"
        return MPS(ss, n -> (mod1(n, period) == 1 ? "Dn" : "Up"))
    else
        error("Invalid initial state! Use 'Up' or 'Dn'.")
    end
end

function make_unitaries(ss::Vector{<:Index}, dt::AbstractFloat)
    lsize = length(ss)
    UA = Vector{ITensor}()
    UB = Vector{ITensor}()
    UC = Vector{ITensor}()

    X = [0 1; 1 0]
    P = [0 0; 0 1]

    pxpm = kron(P, X, P)
    pxpexpm1 = cis(-dt * pxpm)
    pxpexpm2 = cis(-dt/2 * pxpm)
    xpexp = cis( -dt / 2 * kron(X, P))
    pxexp = cis( -dt * kron(P, X))

    for j in 1:(lsize - 2)
        if mod1(j, 3) == 1
            uj = op(pxpexpm2, ss[j], ss[j + 1], ss[j + 2])
            push!(UA, uj)
        elseif mod1(j, 3) == 2
            uj = op(pxpexpm2, ss[j], ss[j + 1], ss[j + 2])
            push!(UB, uj)
        else
            uj = op(pxpexpm1, ss[j], ss[j + 1], ss[j + 2])
            push!(UC, uj)
        end
    end
    push!(UB, op(pxexp, ss[end - 1], ss[end]))
    push!(UC, op(xpexp, ss[1], ss[2]))
    return UA, reverse(UB), UC
end

function tebd_pxp!(psi::MPS, finaltime::Real, nsteps::Int; maxdim::Int=400, cutoff::Real=1e-14)
    ss = siteinds(psi)
    dt = finaltime / nsteps
    UA, UB, UC = make_unitaries(ss, dt)

    for t in 1:nsteps
        psi = apply(UA, psi; cutoff=cutoff, maxdim=maxdim)
        psi = apply(UB, psi; cutoff=cutoff, maxdim=maxdim)
        psi = apply(UC, psi; cutoff=cutoff, maxdim=maxdim)
        psi = apply(UB, psi; cutoff=cutoff, maxdim=maxdim)
        psi = apply(UA, psi; cutoff=cutoff, maxdim=maxdim)

        normalize!(psi)
    end
    return psi
end

function tebd_pxp!(psi::MPS, finaltime::Real, nsteps::Int, obs::AbstractObserver; maxdim::Int=400, cutoff::Real=1e-14)
    ss = siteinds(psi)
    dt = finaltime / nsteps
    UA, UB, UC = make_unitaries(ss, dt)

    mps_record!(obs, psi, 0)
    for t in 1:nsteps
        psi = apply(UA, psi; maxdim=maxdim, cutoff=cutoff)
        psi = apply(UB, psi; maxdim=maxdim, cutoff=cutoff)
        psi = apply(UC, psi; maxdim=maxdim, cutoff=cutoff)
        psi = apply(UB, psi; maxdim=maxdim, cutoff=cutoff)
        psi = apply(UA, psi; maxdim=maxdim, cutoff=cutoff)

        mps_record!(obs, psi, t)
        normalize!(psi)
    end
    return psi
end

let 
    L, nsteps = 24, 600
    tf = 30.0
    ts = range(0.0, tf, nsteps + 1)

    ss = siteinds("S=1/2", L)
    psi = make_initialstate(ss, 2, "Up")
    obs = MyObserver()
    tebd_pxp!(psi, tf, nsteps, obs; maxdim=400)

    fig = Figure()
    ax = Axis(fig[1, 1], xlabel="Time step", ylabel="Correlation")
    lines!(ax, ts, obs.correlations)
    fig
end
