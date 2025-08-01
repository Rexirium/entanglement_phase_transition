using LinearAlgebra
using ITensors, ITensorMPS
include("entanglement_entropies.jl")

function ITensors.op(::OpName"RdU", ::SiteType"S=1/2", s::Index...; eltype=ComplexF64)
    d = prod(dim.(s))
    M = randn(eltype, d, d)
    Q, _ = NDTensors.qr_positive(M)
    return op(Q, s...)
end

function ITensors.op(::OpName"NH", ::SiteType"S=1/2", s::Index; eta::Real)
    return op([1 0; 0 eta], s)
end

function unitaryGenerator(s::Index...; eltype=ComplexF64)
    d = prod(dim.(s))
    M = randn(eltype, d, d)
    Q, _ = NDTensors.qr_positive(M)
    return op(Q, s1, s2)
end

function nonHermitianGenerator(eta::Float64, s0::Index)
    M = [1 0 ; 0 eta]
    return op(M, s0)
end

function mps_evolve(psi0::MPS, ttotal::Int, prob::Real, eta::Real; cutoff::Real=1e-14)
    psi = copy(psi0)
    sites = siteinds(psi)
    Ls= length(sites)
    for t in 1:ttotal
        start = isodd(t) ? 1 : 2
        for j in start:2:Ls-1
            s1, s2 = sites[j], sites[j+1]
            U = op("RdU", s1, s2)
            psi = apply(U, psi; cutoff)
        end
        for j in 1:Ls
            p = rand()
            if p < prob
                s = sites[j]
                M = op("NH", s; eta=eta)
                psi = apply(M, psi; cutoff)
            end
        end
        normalize!(psi)
    end
    return psi
end

function entropy_evolve(psi0::MPS, ttotal::Int, prob::Real, eta::Real, b::Int, which_ent::Real; 
     cutoff::Real=1e-14, ent_cutoff::Real=1e-12)
    psi = copy(psi0)
    sites = siteinds(psi) 
    Ls = length(sites)
    entropies = Float64[]
    ini_entropy = Renyi_entropy(psi0, b, which_ent; cutoff=ent_cutoff)
    push!(entropies, ini_entropy)
    for t in 1:ttotal
        start = isodd(t) ? 1 : 2
        for j in start:2:Ls-1
            s1, s2 = sites[j], sites[j+1]
            U = op("RdU", s1, s2)
            psi = apply(U, psi; cutoff)
        end
        for j in 1:Ls
            samp = rand()
            if samp < prob
                s = sites[j]
                M = op("NH", s; eta=eta)
                psi = apply(M, psi; cutoff)
            end
        end
        normalize!(psi)
        entropy = Renyi_entropy(psi, b, which_ent; cutoff=ent_cutoff)
        push!(entropies, entropy)
    end
    return psi, entropies
end
#=
let 
    L = 8
    T = 2L
    b = L÷2
    p, eta = 0.5, 0.5
    ss = siteinds("S=1/2", L)
    psi0 = MPS(ss, "Up")
    psi, entropies = entropy_evolve(psi0, T, p, eta, b, 1)
    println(entropies)
end
=#
