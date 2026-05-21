using MKL
using ITensors, ITensorMPS
if !isdefined(Main, :RandomUnitary)
    include("../src/RandomUnitary.jl")
    using .RandomUnitary: applyn!, ent_entropy, correlation
end
using CairoMakie

mutable struct MyObserver <: AbstractObserver
    initial::MPS
    steps_per_snapshot::Int
    entropies::Vector{Float64}
    correlations::Vector{Float64}
    overlaps::Vector{Float64}
    maxbonds::Vector{Int}
    truncerrs::Vector{Float64}

    MyObserver(initial::MPS, p::Int) = new(initial, p, Float64[], Float64[], Float64[], Int[], Float64[])
end


function mps_record!(obs::MyObserver, psi::MPS, t::Int)
    if mod(t, obs.steps_per_snapshot) != 0
        return
    end

    b = length(psi) ÷ 2
    orthogonalize!(psi, b)
    push!(obs.entropies, ent_entropy(psi, b))
    push!(obs.correlations, correlation(psi, "Z", "Z", b, b + 1; ortho=true))
    orthogonalize!(psi, 1)
    push!(obs.overlaps, abs2(inner(obs.initial, psi)))
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

    truncerrs = Float64[0.0]
    truncerr = 0.0

    mps_record!(obs, psi, 0)
    for t in 1:nsteps
        truncerr += applyn!(UA, psi; maxdim=maxdim, cutoff=cutoff)
        truncerr += applyn!(UB, psi; maxdim=maxdim, cutoff=cutoff, rev=true)
        truncerr += applyn!(UC, psi; maxdim=maxdim, cutoff=cutoff)
        truncerr += applyn!(UB, psi; maxdim=maxdim, cutoff=cutoff, rev=true)
        truncerr += applyn!(UA, psi; maxdim=maxdim, cutoff=cutoff)

        mps_record!(obs, psi, t)
        push!(truncerrs, truncerr)
        normalize!(psi)
    end
    return psi, truncerrs
end

function tebd_pxp!(psi::MPS, finaltime::Real, nsteps::Int, obs::AbstractObserver; maxdim::Int=400, cutoff::Real=1e-14)
    ss = siteinds(psi)
    dt = finaltime / nsteps
    UA, UB, UC = make_unitaries(ss, dt)
    reverse!(UB)

    truncerrs = Float64[0.0]
    truncerr = 0.0

    mps_record!(obs, psi, 0)
    for t in 1:nsteps
        truncerr += applyn!(UA, psi; maxdim=maxdim, cutoff=cutoff)
        truncerr += applyn!(UB, psi; maxdim=maxdim, cutoff=cutoff, rev=true)
        truncerr += applyn!(UC, psi; maxdim=maxdim, cutoff=cutoff)
        truncerr += applyn!(UB, psi; maxdim=maxdim, cutoff=cutoff, rev=true)
        truncerr += applyn!(UA, psi; maxdim=maxdim, cutoff=cutoff)

        mps_record!(obs, psi, t)
        push!(truncerrs, truncerr)
        normalize!(psi)
    end
    return truncerrs
end

function tdvp_pxp!(psi::MPS, finaltime::Real, nsteps::Int, obs::AbstractObserver; maxdim::Int=400, cutoff::Real=1e-14, krylovdim::Int=16)
    ss = siteinds(psi)
    H = make_Hamiltonian(ss)

    dt = finaltime / nsteps

    mps_record!(obs, psi, 0)
    for t in 1:nsteps
        psi = tdvp(H, -im * dt, psi; nsteps=1, 
            maxdim=maxdim, cutoff=cutoff, 
            updater_kwargs=(; tol=1e-4, krylovdim=krylovdim), 
            outputlevel=0)
        mps_record!(obs, psi, t)
    end
end

let 
    L, nsteps = 36 , 600
    b = L ÷ 2

    tf = 30.0
    p = 3
    ts = range(0.0, tf, nsteps ÷ p + 1)

    ss = siteinds("S=1/2", L)
    psi = make_initialstate(ss, 2, "Up")
    H = make_Hamiltonian(ss)

    obs = MyObserver(copy(psi), p)

    @time errs = tebd_pxp!(psi, tf, nsteps, obs; maxdim=256, cutoff=1e-12)

    fig = Figure(size=(800, 600))
    ax1 = Axis(fig[1, 1], xlabel="t", ylabel="Entropy")
    lines!(ax1, ts, obs.entropies)

    ax2 = Axis(fig[1, 2], xlabel="t", ylabel="Correlation")
    lines!(ax2, ts, obs.correlations)

    ax3 = Axis(fig[2, 1], xlabel="t", ylabel="Overlap")
    lines!(ax3, ts, obs.overlaps)

    ax4 = Axis(fig[2, 2], xlabel="t", ylabel="Truncation Error")
    lines!(ax4, ts, errs[1 : p : end])
    fig
    #save("pxp_tebd_results.png", fig)
    
end
