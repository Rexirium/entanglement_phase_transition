using MKL, LinearAlgebra
using ITensors, ITensorMPS
include("../src/entanglement.jl")
using CairoMakie

mutable struct MyObserver <: AbstractObserver
    entropies::Vector{Float64}
    maxbonds::Vector{Int}

    MyObserver() = new(Float64[], Int[])
end

function mps_record!(obs::MyObserver, psi::MPS, t::Int)
    push!(obs.entropies, ent_entropy(psi, length(psi) ÷ 2))
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

    pxpm = kron([0 0; 0 1], [0 1; 1 0], [0 0; 0 1])
    pxpexpm1 = cis(-dt * pxpm)
    pxpexpm2 = cis(-dt/2 * pxpm)

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
    L, nsteps = 24, 200
    tf = 20.0
    ss = siteinds("S=1/2", L)
    psi = make_initialstate(ss, 3, "Up")
    obs = MyObserver()
    tebd_pxp!(psi, tf, nsteps, obs; maxdim=400)

    fig = Figure()
    ax = Axis(fig[1, 1], xlabel="Time step", ylabel="Entropy")
    lines!(ax, 0:nsteps, obs.entropies)
    fig
end
