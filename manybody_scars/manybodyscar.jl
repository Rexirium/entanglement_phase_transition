using MKL
using ITensors, ITensorMPS
if !isdefined(Main, :RandomUnitary)
    include("../src/RandomUnitary.jl")
    using .RandomUnitary: applyn!, InfMPS, ent_entropy, correlation
end

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
        orthogonalize!(psi, 1)
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

function mps_record!(obs::MyObserver, psi::InfMPS, initial::InfMPS, t::Int, err::Float64)
    if mod(t, obs.steps_per_snapshot) != 0
        orthogonalize!(psi, 1)
        return
    end

    push!(obs.entropies, ent_entropy(psi))
    push!(obs.correlations, correlation(psi, "Z", "Z"; ortho=true))
    push!(obs.overlaps, abs2(inner(initial, psi)))
    push!(obs.maxbonds, maxlinkdim(psi))
    push!(obs.truncerrs, err)
end

function make_initialstate(ss::Vector{<:Index}, period::Int, start::String)
    if period == 1
        return MPS(ss, n -> "Dn")
    end

    if start == "Up"
        return MPS(ss, n -> (mod(n, period) == 1 ? "Up" : "Dn"))
    elseif start == "Dn"
        return MPS(ss, n -> (mod(n, period) == 0 ? "Up" : "Dn"))
    else
        error("Invalid initial state! Use 'Up' or 'Dn'.")
    end
end

function make_initialinfstate(ss::Vector{<:Index}, period::Int, start::String)
    if period == 1
        return InfMPS(ss, n -> "Dn")
    end

    if start == "Up"
        return InfMPS(ss, n -> (mod(n, period) == 1 ? "Up" : "Dn"))
    elseif start == "Dn"
        return InfMPS(ss, n -> (mod(n, period) == 0 ? "Up" : "Dn"))
    else
        error("Invalid initial state! Use 'Up' or 'Dn'.")
    end
end

function make_unitaries(psi::MPS, dt::AbstractFloat)
    lsize = length(psi)
    ss = siteinds(psi)
    UA = Vector{ITensor}()
    UB = Vector{ITensor}()
    UC = Vector{ITensor}()

    X = Int64[0 1; 1 0]
    P = Int64[0 0; 0 1]

    pxpm = kron(P, X, P)
    pxpexpm1 = cis(-dt * pxpm)
    pxpexpm2 = cis(-dt/2 * pxpm)

    xpexp = cis( -dt * kron(X, P))
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
        pxexp =  cis(-dt * kron(P, X))
        push!(UC, op(pxexp, ss[end - 1], ss[end]))
    elseif mod(lsize, 3) == 2
        pxexp =  cis(-dt/2 * kron(P, X))
        push!(UA, op(pxexp, ss[end - 1], ss[end]))
    else
        pxexp =  cis(-dt/2 * kron(P, X))
        push!(UB, op(pxexp, ss[end - 1], ss[end]))
    end

    return UA, reverse(UB), UC
end

function make_unitaries(psi::InfMPS, dt::AbstractFloat)
    len_uc = psi.len_uc
    ss = siteinds(psi)

    UA = Vector{ITensor}()
    UB = Vector{ITensor}()
    UC = Vector{ITensor}()

    X = Int64[0 1; 1 0]
    P = Int64[0 0; 0 1]

    pxpm = kron(P, X, P)
    pxpexpm1 = cis(-dt * pxpm)
    pxpexpm2 = cis(-dt/2 * pxpm)

    for j in 1 : len_uc
        nj = mod1(j + 1, len_uc)
        nnj = mod1(j + 2, len_uc)

        if mod(j, 3) == 1
            push!(UA, op(pxpexpm2, ss[j], ss[nj], ss[nnj]))
        elseif mod(j, 3) == 2
            push!(UB, op(pxpexpm2, ss[j], ss[nj], ss[nnj]))
        else
            push!(UC, op(pxpexpm1, ss[j], ss[nj], ss[nnj]))
        end
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

function tebd_pxp(psi0::Union{MPS, InfMPS}, finaltime::Real, nsteps::Int, obs::AbstractObserver; maxdim::Int=400, cutoff::Real=1e-12, etol=nothing)
    psi = copy(psi0)
    dt = finaltime / nsteps
    UA, UB, UC = make_unitaries(psi, dt)

    truncerr = 0.0
    mps_record!(obs, psi, psi0, 0, truncerr)
    for t in 1:nsteps
        truncerr += applyn!(UA, psi; maxdim=maxdim, cutoff=cutoff)
        truncerr += applyn!(UB, psi; maxdim=maxdim, cutoff=cutoff, rev=true)
        normalize!(psi)
        truncerr += applyn!(UC, psi; maxdim=maxdim, cutoff=cutoff)
        normalize!(psi)
        truncerr += applyn!(UB, psi; maxdim=maxdim, cutoff=cutoff, rev=true)
        truncerr += applyn!(UA, psi; maxdim=maxdim, cutoff=cutoff)
        normalize!(psi)

        mps_record!(obs, psi, psi0, t, truncerr)

        if !isnothing(etol) && truncerr > etol
            println("Early stop at step $t (out of $nsteps) due to truncation error exceeding etol.")
            println(": $truncerr > $etol")
            break
        end
    end
    return psi, truncerr
end

function tebd_pxp!(psi::Union{MPS, InfMPS}, finaltime::Real, nsteps::Int, obs::AbstractObserver; maxdim::Int=400, cutoff::Real=1e-12, etol=nothing)
    initial = copy(psi)
    dt = finaltime / nsteps
    UA, UB, UC = make_unitaries(psi, dt)

    truncerr = 0.0
    mps_record!(obs, psi, initial, 0, truncerr)
    for t in 1:nsteps
        truncerr += applyn!(UA, psi; maxdim=maxdim, cutoff=cutoff)
        truncerr += applyn!(UB, psi; maxdim=maxdim, cutoff=cutoff, rev=true)
        normalize!(psi)
        truncerr += applyn!(UC, psi; maxdim=maxdim, cutoff=cutoff)
        normalize!(psi)
        truncerr += applyn!(UB, psi; maxdim=maxdim, cutoff=cutoff, rev=true)
        truncerr += applyn!(UA, psi; maxdim=maxdim, cutoff=cutoff)
        normalize!(psi)

        mps_record!(obs, psi, initial, t, truncerr)

        if !isnothing(etol) && truncerr > etol
            println("Early stop at step $t (out of $nsteps) due to truncation error exceeding etol.")
            println(": $truncerr > $etol")
            break
        end
    end
    return truncerr
end



