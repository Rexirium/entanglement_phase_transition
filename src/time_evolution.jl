using ITensors, ITensorMPS
using LinearAlgebra, Random
using SparseArrays

include("entanglement.jl")
include("correlation.jl")
include("observers.jl")

const CNOT13 = begin
    X = sparse([0 1; 1 0])
    Id = sparse(I, 4, 4)
    blockdiag(Id, X, X)
end

abstract type AbstractDisentangler end

struct NHDisentangler{Tp <: Real} <: AbstractDisentangler
    prob::Tp
    eta::Tp
end

struct NHCNOTDisentangler{Tp <: Real} <: AbstractDisentangler
    prob::Tp
    eta::Tp
end


function ITensors.op(::OpName"RdU", ::SiteType"S=1/2", s::Index...; eltype::DataType=ComplexF64)
    """
    Create a random unitary operator for the given site indices `s`.
    """
    M = randn(eltype, 4, 4)
    Q, _ = NDTensors.qr_positive(M)
    return op(Q, s...)
end

function ITensors.op(::OpName"NH", ::SiteType"S=1/2", s::Index; eta::Real)
    """
    Create a non-Hermitian operator for the given site index `s` with parameter `eta`.
    """
    M = diagm(shuffle([one(eta), eta]))
    return op(M, s)
end

function ITensors.op(::OpName"NHCNOT", ::SiteType"S=1/2", s::Index...; eta::Real)
    """
    Create a CNOT-based non-Hermitian operator for the given site indices `s1` and `s2` with parameter `eta`.
    """
    id = ones(typeof(eta), 2)
    dd = kron(id, shuffle([one(eta), eta]), id)
    return op(diagm(dd) * CNOT13, s...)
end

function ITensors.op(::OpName"WM", ::SiteType"S=1/2", s::Index; x::Real, λ::Real=1.0, Δ::Real=1.0)
    """Create a weak measurement operator for the given site index `s` with parameters `x`, `λ`, and `Δ`."""
    # Assuming `x` is a random variable from a Gaussian distribution
    phiUp = exp(-(x-λ)*(x-λ) / (4*Δ*Δ))
    phiDn = exp(-(x+λ)*(x+λ) / (4*Δ*Δ))
    return op([phiUp 0; 0 phiDn], s)
end

function proj_measure!(psi::MPS, loc::Int)
    """Perform a projective measurement on the MPS `psi` at site `loc` with outcome `:Up` or `:Dn`."""
    (loc <= 0 || loc > length(psi)) && return psi
    # Orthogonalize the MPS at site `loc`
    s = siteind(psi, loc)
    projUp = op("ProjUp", s)
    orthogonalize!(psi, loc)
    # Calculate the probability of measuring "Up"
    probUp = inner(prime(psi[loc], tags="Site"), projUp, psi[loc]).re
    samp = rand()
    if samp < probUp
        apply1!(projUp, psi, loc)
    else
        projDn = op("ProjDn", s)
        apply1!(projDn, psi, loc)
    end
    normalize!(psi)
end

function weak_measure!(psi::MPS, loc::Int, para::Tuple{T, T}=(1.0, 1.0)) where T<:Real
    """Perform a weak measurement on the MPS `psi` at site `loc` with parameters `λ` and `Δ`."""
    (loc <= 0 || loc > length(psi)) && return psi
    # Orthogonalize the MPS at site `loc`
    s = siteind(psi, loc)
    λ, Δ = para
    
    proj = op("ProjUp", s)
    orthogonalize!(psi, loc)
    # Calculate the probability of measuring "Up"
    probUp = inner(prime(psi[loc], tags="Site"), proj, psi[loc]).re
    samp = rand(T)
    # generate a random variable from a Gaussian distribution
    x = samp < probUp ? λ + Δ*randn(T) : -λ + Δ*randn(T)
    M = op("WM", s; x = x, λ = λ, Δ = Δ)
    # Apply the weak measurement operator
    apply1!(M, psi, loc)
    normalize!(psi)
end

function apply1!(G1::ITensor, psi::MPS, loc::Int)
    """
    Apply the gate `G1` to the MPS `psi` at site `loc` inplace.
    """
    orthogonalize!(psi, loc)
    A = noprime(psi[loc] * G1)
    psi[loc] = A
end

function apply2!(G2::ITensor, psi::MPS, j1::Int; cutoff::Real=1e-14, maxdim::Int=2*maxlinkdim(psi))
    """
    Apply two adjacent site gate `G2` to the MPS `psi` at sites `j1` and `j1+1` inplace.
    """
    (j1<=0 || j1>= length(psi)) && error("Wrong starting site for two-site gate application.")
    orthogonalize!(psi, j1)
    j2 = j1 + 1
    A = (psi[j1] * psi[j2]) * G2
    noprime!(A)
    linds = uniqueinds(psi[j1], psi[j2])
    psi[j1], S, psi[j2], spec = svd(A, linds; cutoff=cutoff, maxdim=maxdim)
    psi[j2] *= S
    set_ortho_lims!(psi, j2:j2)
    return spec.truncerr
end

function apply3!(G3::ITensor, psi::MPS, j2::Int; cutoff::Real=1e-14, maxdim::Int=2*maxlinkdim(psi))
    """
    Apply three adjacent site gate `G3` to the MPS `psi` at sites `j2-1`, `j2`, and `j2+1` inplace.
    """
    (j2 <= 1 || j2 >= length(psi)) && error("Wrong middle site for three-site gate application.")
    orthogonalize!(psi, j2)
    s = siteind(psi, j2)
    j1, j3 = j2 - 1, j2 + 1
    A = (psi[j1] * psi[j2] * psi[j3]) * G3
    noprime!(A)
    linds = uniqueinds(psi[j1], psi[j2])
    psi[j1], S12, B, spec12 = svd(A, linds; cutoff=cutoff, maxdim=maxdim)
    B *= S12
    linds23 = (commonind(psi[j1], B), s)
    psi[j2], S23, psi[j3], spec23 = svd(B, linds23; cutoff=cutoff, maxdim=maxdim)
    psi[j2] *= S23
    set_ortho_lims!(psi, j2:j2)
    return spec12.truncerr + spec23.truncerr
end

function disentangle!(psi::MPS, dent::NHDisentangler{Tp}) where Tp<:Real
    """
    Apply the non-Hermitian disentangler to the MPS `psi` inplace.
    """
    for j in length(psi):-1:1
        if rand() < dent.prob
            M = op("NH", siteind(psi, j); eta=dent.eta)
            apply1!(M, psi, j)
            normalize!(psi)
        end
    end
    return zero(Tp)
end

function disentangle!(psi::MPS, dent::NHCNOTDisentangler{Tp}) where Tp<:Real
    """
    Apply the CNOT-based non-Hermitian disentangler to the MPS `psi` inplace.
    """
    ss = siteinds(psi)
    truncerr = zero(Tp)
    for j in (length(psi)-1):-1:2
        if rand() < dent.prob
            M = op("NHCNOT", ss[j-1 : j+1]...; eta=dent.eta)
            err += apply3!(M, psi, j)
            truncerr += err
            normalize!(psi)
        end
    end
    return truncerr
end

function mps_evolve!(psi::MPS, ttotal::Int, dent::AbstractDisentangler; 
    cutoff::Real=1e-14, maxdim::Int=1<<(length(psi) ÷ 2), etol=nothing)
    """
    Evolve the MPS `psi0` for `ttotal` time steps with each time step a random unitary operator applied to pairs of sites,
    and a non-Hermitian operator applied to each site with probability `prob` and parameter `eta`. (inplace version)
    """
    sites = siteinds(psi)
    lsize = length(sites)
    T = promote_itensor_eltype(psi)
    
    truncerr = zero(real(T))
    for t in 1:ttotal
        # Apply random unitary operators to pairs of sites
        for j in (iseven(t) + 1):2:lsize-1
            U = op("RdU", sites[j], sites[j+1]; eltype=T)
            err = apply2!(U, psi, j; cutoff=cutoff, maxdim=maxdim)
            truncerr += err
        end
        # Apply non-Hermitian disentanglers
        truncerr += disentangle!(psi, dent)
        # break if truncation error exceeds etol
        if !isnothing(etol) && truncerr > etol
            obs.accept = false
            break
        end
    end
    return truncerr
end

function mps_evolve!(psi::MPS, ttotal::Int, dent::AbstractDisentangler, obs::AbstractObserver; 
    cutoff::Real=1e-14, maxdim::Int=1<<(length(psi) ÷ 2), etol=nothing)
    """
    Evolve the MPS `psi0` for `ttotal` time steps with each time step a random unitary operator applied to pairs of sites,
    and a non-Hermitian operator applied to each site with probability `prob` and parameter `eta`. (inplace version)
    """
    sites = siteinds(psi)
    lsize = length(sites)
    T = promote_itensor_eltype(psi)
    
    truncerr = zero(real(T))
    mps_monitor!(obs, psi, 0, truncerr)
    for t in 1:ttotal
        # Apply random unitary operators to pairs of sites
        for j in (iseven(t) + 1):2:lsize-1
            U = op("RdU", sites[j], sites[j+1]; eltype=T)
            err = apply2!(U, psi, j; cutoff=cutoff, maxdim=maxdim)
            truncerr += err
        end
        # Apply non-Hermitian disentanglers
        truncerr += disentangle!(psi, dent)
        # Monitor the MPS and truncation error
        mps_monitor!(obs, psi, t, truncerr)
        # break if truncation error exceeds etol
        if !isnothing(etol) && truncerr > etol
            obs.accept = false
            break
        end
    end
    return truncerr
end

#=
let 
    L, T = 10, 100
    ss = siteinds("S=1/2", L)
    psi = MPS(ComplexF64, ss, "Up")
    dent = NHDisentangler{Float64}(0.5, 0.5)
    @time mps_evolve!(psi, T, dent; cutoff=1e-14, maxdim=100)

end
=#
