using MKL
using ITensors, ITensorMPS
if !isdefined(Main, :RandomUnitary)
    include("../src/RandomUnitary.jl")
end
using .RandomUnitary: applyn!, ent_entropy, correlation

mutable struct MyObserver <: AbstractObserver
    steps_per_snapshot::Int
    entropies::Vector{Float64}
    correlations::Vector{Float64}
    overlaps::Vector{Float64}
    maxbonds::Vector{Int}
    truncerrs::Vector{Float64}

    MyObserver(p::Int) = new(p, Float64[], Float64[], Float64[], Int[], Float64[])
end


function mps_record!(obs::MyObserver, psi::MPS, initial::MPS, t::Int, err::Float64)
    if mod(t, obs.steps_per_snapshot) != 0
        return
    end

    b = length(psi) ÷ 2
    orthogonalize!(psi, b)
    push!(obs.entropies, ent_entropy(psi, b))
    push!(obs.correlations, correlation(psi, "Z", "Z", b, b + 1; ortho=true))
    orthogonalize!(psi, 1)
    push!(obs.overlaps, abs2(inner(initial, psi)))
    push!(obs.maxbonds, maxlinkdim(psi))
    push!(obs.truncerrs, err)
end

function make_initialstate(ss::Vector{<:Index}, period::Int, start::String)
    if period == 1
        return MPS(ss, n -> "Dn")
    end

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

    X = Int64[0 1; 1 0]
    P = Int64[0 0; 0 1]

    pxpm = kron(P, X, P)
    pxpexpm1 = cis(-dt * pxpm)
    pxpexpm2 = cis(-dt/2 * pxpm)
    xpexp = cis( -dt / 2 * kron(X, P))
    pxexp = cis( -dt * kron(P, X))

    push!(UC, op(xpexp, ss[1], ss[2]))
    for j in 1:(lsize - 2)
        if mod(j, 3) == 1
            uj = op(pxpexpm2, ss[j], ss[j + 1], ss[j + 2])
            push!(UA, uj)
        elseif mod(j, 3) == 2
            uj = op(pxpexpm2, ss[j], ss[j + 1], ss[j + 2])
            push!(UB, uj)
        else
            uj = op(pxpexpm1, ss[j], ss[j + 1], ss[j + 2])
            push!(UC, uj)
        end
    end

    if mod(lsize, 3) == 1
        push!(UC, op(pxexp, ss[end - 1], ss[end]))
    elseif mod(lsize, 3) == 2
        push!(UA, op(pxexp, ss[end - 1], ss[end]))
    else
        push!(UB, op(pxexp, ss[end - 1], ss[end]))
    end

    return UA, UB, UC
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

function tebd_pxp(psi0::MPS, finaltime::Real, nsteps::Int, obs::AbstractObserver; maxdim::Int=400, cutoff::Real=1e-14)
    psi = copy(psi0)
    ss = siteinds(psi)
    dt = finaltime / nsteps
    UA, UB, UC = make_unitaries(ss, dt)
    reverse!(UB)

    truncerr = 0.0
    mps_record!(obs, psi, psi0, 0, truncerr)
    for t in 1:nsteps
        truncerr += applyn!(UA, psi; maxdim=maxdim, cutoff=cutoff)
        truncerr += applyn!(UB, psi; maxdim=maxdim, cutoff=cutoff, rev=true)
        truncerr += applyn!(UC, psi; maxdim=maxdim, cutoff=cutoff)
        truncerr += applyn!(UB, psi; maxdim=maxdim, cutoff=cutoff, rev=true)
        truncerr += applyn!(UA, psi; maxdim=maxdim, cutoff=cutoff)

        mps_record!(obs, psi, psi0, t, truncerr)
        normalize!(psi)
    end
    return psi, truncerr
end

function tebd_pxp!(psi::MPS, finaltime::Real, nsteps::Int, obs::AbstractObserver; maxdim::Int=400, cutoff::Real=1e-14)
    initial = copy(psi)
    ss = siteinds(psi)
    dt = finaltime / nsteps
    UA, UB, UC = make_unitaries(ss, dt)
    reverse!(UB)

    truncerr = 0.0
    mps_record!(obs, psi, initial, 0, truncerr)
    for t in 1:nsteps
        truncerr += applyn!(UA, psi; maxdim=maxdim, cutoff=cutoff)
        truncerr += applyn!(UB, psi; maxdim=maxdim, cutoff=cutoff, rev=true)
        truncerr += applyn!(UC, psi; maxdim=maxdim, cutoff=cutoff)
        truncerr += applyn!(UB, psi; maxdim=maxdim, cutoff=cutoff, rev=true)
        truncerr += applyn!(UA, psi; maxdim=maxdim, cutoff=cutoff)

        mps_record!(obs, psi, initial, t, truncerr)
        normalize!(psi)
    end
    return truncerr
end
#=
function tdvp_pxp!(psi::MPS, finaltime::Real, nsteps::Int, obs::AbstractObserver; maxdim::Int=400, cutoff::Real=1e-14, krylovdim::Int=16)
    initial = copy(psi)
    ss = siteinds(psi)
    H = make_Hamiltonian(ss)

    dt = finaltime / nsteps

    mps_record!(obs, psi, initial, 0, 0.0)
    for t in 1:nsteps
        psi = tdvp(H, -im * dt, psi; nsteps=1, 
            maxdim=maxdim, cutoff=cutoff, 
            updater_kwargs=(; tol=1e-4, krylovdim=krylovdim), 
            outputlevel=0)
        mps_record!(obs, psi, initial, t, 0.0)
    end
    return 0.0
end
=#
function main(lsize::Int, period::Int)
    ss = siteinds("S=1/2", lsize)
    psi = make_initialstate(ss, period, "Up")
    
    obs = MyObserver(3)
    tebd_pxp!(psi, 30.0, 600, obs; maxdim=256, cutoff=1e-12)
    return obs
end

let 
    L = 18

    @time obs = main(L, 4)

    obs.entropies
end

