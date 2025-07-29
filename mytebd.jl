using ITensors, ITensorMPS
using BenchmarkTools

function makeLayer(s::Vector{<:Index}, tau::Real)
    Ls = length(s)
    gates = ITensor[]
    for j in 1:Ls-1
        s1, s2 = s[j], s[j+1]
        hj = op("Sz", s1) * op("Sz", s2) +
              0.5 * op("S+", s1) * op("S-", s2) + 
              0.5 * op("S-", s1) * op("S+", s2)
        Gj = exp(- im * tau/2 * hj)
        push!(gates, Gj)
    end
    append!(gates, reverse(gates))
    return gates
end

function makeoddLayer(s::Vector{<:Index}, tau::Real)
    Ls = length(s)
    gates = ITensor[]
    for j in 1:2:Ls-1
        s1, s2 = s[j], s[j+1]
        hj = op("Sz", s1) * op("Sz", s2) +
              0.5 * op("S+", s1) * op("S-", s2) + 
              0.5 * op("S-", s1) * op("S+", s2)
        Gj = exp(- im * tau * hj)
        push!(gates, Gj)
    end
    return gates
end

function makeevenLayer(s::Vector{<:Index}, tau::Real)
    Ls = length(s)
    gates = ITensor[]
    for j in 2:2:Ls-1
        s1, s2 = s[j], s[j+1]
        hj = op("Sz", s1) * op("Sz", s2) +
              0.5 * op("S+", s1) * op("S-", s2) + 
              0.5 * op("S-", s1) * op("S+", s2)
        Gj = exp(- im * tau * hj)
        push!(gates, Gj)
    end
    return gates
end

function timeEvolve(psi0::MPS, ttotal::Real, maxiter::Int; cutoff::Real=1e-14)
    tau = ttotal / maxiter
    psi = copy(psi0)
    gates = makeLayer(siteinds(psi), tau)
    for iter in 1:maxiter
        psi = apply(gates, psi; cutoff)
        normalize!(psi)
    end
    return psi
end

function timeEvolve2(psi0::MPS, ttotal::Real, maxiter::Int; cutoff::Real=1e-14)
    tau = ttotal / maxiter
    psi = copy(psi0)
    gates1 = makeoddLayer(siteinds(psi), tau)
    gates2 = makeevenLayer(siteinds(psi), tau)
    for iter in 1:maxiter
        psi = apply(gates1, psi; cutoff)
        # normalize!(psi)
        psi = apply(gates2, psi; cutoff)
        normalize!(psi)
    end
    return psi
end

function timeEvolve3(psi0::MPS, ttotal::Real, maxiter::Int; cutoff::Real=1e-14)
    tau = ttotal / maxiter
    psi = copy(psi0)
    sites = siteinds(psi)
    Ls= length(sites)
    for iter in 1:maxiter
        for j in 1:2:Ls-1
            s1, s2 = sites[j], sites[j+1]
            hj = op("Sz", s1) * op("Sz", s2) +
              0.5 * op("S+", s1) * op("S-", s2) + 
              0.5 * op("S-", s1) * op("S+", s2)
            Gj = exp(- im * tau * hj)
            psi = apply(Gj, psi; cutoff)
        end
        # normalize!(psi)
        for j in 2:2:Ls-1
            s1, s2 = sites[j], sites[j+1]
            hj = op("Sz", s1) * op("Sz", s2) +
              0.5 * op("S+", s1) * op("S-", s2) + 
              0.5 * op("S-", s1) * op("S+", s2)
            Gj = exp(- im * tau * hj)
            psi = apply(Gj, psi; cutoff)
        end
        normalize!(psi)
    end
    return psi
end

let
    L, N = 10, 40
    T = 1.0
    sites = siteinds("S=1/2", L)
    psi0 = MPS(sites, n -> isodd(n) ? "Up" : "Dn")
    @benchmark timeEvolve3($psi0, $T, $N; cutoff=1e-12)
end
