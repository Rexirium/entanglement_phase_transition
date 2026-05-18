using MKL
using ITensors, ITensorMPS
using Observers: observer
include("../src/entanglement.jl")
include("../src/correlation.jl")
using CairoMakie

mutable struct MyObserver <: AbstractObserver
    entropies::Vector{Float64}
    correlations::Vector{Float64}
    maxbonds::Vector{Int}

    MyObserver() = new(Float64[], Float64[], Int[])
end

mutable struct DmObserver <: AbstractObserver
    maxbonds::Vector{Int}

    DmObserver() = new(Int[])
end

function mps_record!(obs::MyObserver, psi::MPS, t::Int)
    b = length(psi) ÷ 2
    orthogonalize!(psi, b)
    push!(obs.entropies, ent_entropy(psi, b))
    push!(obs.correlations, correlation(psi, "Z", "Z", b, b + 1; ortho=true))
    push!(obs.maxbonds, maxlinkdim(psi))
    orthogonalize!(psi, 1)
end

function mps_record!(obs::DmObserver, psi::MPS, t::Int)
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

    push!(UC, op(xpexp, ss[1], ss[2]))
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
    return UA, reverse(UB), UC
end

function make_Hamiltonian(ss::Vector{<:Index})
    lsize = length(ss)

    os = OpSum()
    os += "X", 1, "ProjDn", 2
    for j in 1:(lsize - 2)
        os += "ProjDn", j, "X", j + 1, "ProjDn", j + 2
    end
    os += "ProjDn", lsize - 1, "X", lsize

    return MPO(os, ss)
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
#=
function tdvp_pxp!(psi::MPS, finaltime::Real, nsteps::Int, obs; maxdim::Int=400, cutoff::Real=1e-14)
    ss = siteinds(psi)
    H = make_Hamiltonian(ss)

    psi = tdvp(H, -im * finaltime, psi; nsteps=nsteps, maxdim=maxdim, cutoff=cutoff, (step_observer!)=obs, outputlevel=0)
    return psi
end
=#
function tdvp2_pxp!(psi::MPS, finaltime::Real, nsteps::Int, obs::AbstractObserver; maxdim::Int=400, cutoff::Real=1e-14)
    ss = siteinds(psi)
    H = make_Hamiltonian(ss)

    dt = finaltime / nsteps
    mps_record!(obs, psi, 0)
    for t in 1:nsteps
        psi = tdvp(H, -im * dt, psi; nsteps=1, maxdim=maxdim, cutoff=cutoff, outputlevel=0)
        mps_record!(obs, psi, t)
    end
    
    return psi
end

let 
    L, nsteps = 24, 20
    b = L ÷ 2
    tf = 10.0
    ts = range(0.0, tf, nsteps + 1)

    ss = siteinds("S=1/2", L)
    psi = make_initialstate(ss, 2, "Up")
    H = make_Hamiltonian(ss)

    get_entropy(; state) = ent_entropy(state, b)
    get_maxbond(; state) = maxlinkdim(state)
    get_correlation(; state) = begin
        corr = correlation(state, "Z", "Z", b, b + 1)
        orthogonalize!(state, 1)
        corr
    end

    obs = observer("entropies"=>get_entropy, "correlations"=>get_correlation, "maxbonds"=>get_maxbond)
    obs2 = MyObserver()

    @time psi = tdvp2_pxp!(psi, tf, nsteps, obs2; maxdim=400, cutoff=1e-14)

    fig = Figure()
    ax = Axis(fig[1, 1], xlabel="Time step", ylabel="Correlation")
    lines!(ax, ts[2:end], obs2.correlations)
    fig
    
end
