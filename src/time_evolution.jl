using ITensors, ITensorMPS
using LinearAlgebra, Random
include("entanglement.jl")
include("correlation.jl")

struct CalcResult{T}
    entr_mean::T
    entr_std::T
    corr_mean::Vector{T}
    corr_std::Vector{T}
end

mutable struct EntropyObserver{T} <: AbstractObserver
    """
    Observe and record the entanglement entropy at a specific bond `b` of the MPS during time evolution.
    """
    b::Int
    n::Real
    entropies::Vector{T}
    truncerrs::Vector{T}
    maxbonds::Vector{Int}

    EntropyObserver{T}(b::Int; n::Real=1) where T<:Real = new{T}(b, n, T[], T[], Int[])
end

mutable struct EntrCorrObserver{T} <: AbstractObserver
    b::Int
    len::Int
    n::Real
    op::String
    entrs::Vector{T}
    corrs::Vector{Vector{T}}
    truncerrs::Vector{T}

    EntrCorrObserver{T}(b::Int, len::Int; n::Real=1, op::String="Sz") where T<:Real = 
        new{T}(b, len, n, op, T[], Vector{T}[], T[])
end

mutable struct EntropyAverager{T} <: AbstractObserver
    b::Int
    len::Int
    n::Real
    entr_mean::T
    entr_sstd::T

    EntropyAverager{T}(b::Int, len::Int; n::Real=1) where T<:Real = 
        new{T}(b, len, n, zero(T), zero(T))
end

mutable struct EntrCorrAverager{T} <: AbstractObserver
    b::Int
    len::Int
    n::Real
    op::String
    entr_mean::T
    entr_sstd::T
    corr_mean::Vector{T}
    corr_sstd::Vector{T}

    EntrCorrAverager{T}(b::Int, len::Int; n::Real=1, op::String="Sz") where T<:Real = 
        new{T}(b, len, n, op, zero(T), zero(T), 
        zeros(T, len), zeros(T, len))
end

function ITensors.op(::OpName"RdU", ::SiteType"S=1/2", s::Index...; eltype::DataType=ComplexF64)
    """
    Create a random unitary operator for the given site indices `s`.
    """
    M = randn(eltype, 4, 4)
    Q, _ = NDTensors.qr_positive(M)
    return op(Q, s...)
end

function ITensors.op(::OpName"NH", ::SiteType"S=1/2", s::Index; eta::T) where T<:Real
    """
    Create a non-Hermitian operator for the given site index `s` with parameter `eta`.
    """
    M = diagm(shuffle([1, eta]))
    return op(M, s)
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
    probUp = real(inner(prime(psi[loc], tags="Site"), projUp, psi[loc]))
    samp = rand()
    if samp < probUp
        apply!(projUp, psi, loc)
    else
        projDn = op("ProjDn", s)
        apply!(projDn, psi, loc)
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
    probUp = real(inner(prime(psi[loc], tags="Site"), proj, psi[loc]))
    samp = rand(T)
    # generate a random variable from a Gaussian distribution
    x = samp < probUp ? λ + Δ*randn(T) : -λ + Δ*randn(T)
    M = op("WM", s; x = x, λ = λ, Δ = Δ)
    # Apply the weak measurement operator
    apply!(M, psi, loc)
    normalize!(psi)
end

function apply!(G1::ITensor, psi::MPS, loc::Int)
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

function mps_evolve(psi0::MPS, ttotal::Int, prob::Tp, eta::Real; 
    cutoff::Real=1e-14, maxdim::Int=2<<length(psi0), etol=nothing) where Tp<:Real
    """
    Evolve the MPS `psi0` for `ttotal` time steps with each time step a random unitary operator applied to pairs of sites,
    and a non-Hermitian operator applied to each site with probability `prob` and parameter `eta`.
    """
    psi = copy(psi0)
    T = promote_itensor_eltype(psi)
    sites = siteinds(psi)
    lsize = length(sites)
    
    truncerr = 0.0
    for t in 1:ttotal
        # Apply random unitary operators to pairs of sites
        for j in (iseven(t) + 1):2:lsize-1
            U = op("RdU", sites[j], sites[j+1]; eltype=T)
            err = apply2!(U, psi, j; cutoff=cutoff, maxdim=maxdim)
            truncerr += err
        end
        # Apply non-Hermitian operator to each site with probability `prob` and parameter `eta`
        for j in 1:lsize
            if rand(Tp) >= prob
                continue
            end
            M = op("NH", sites[j]; eta=eta)
            apply!(M, psi, j)
            # Normalize the MPS after applying the non Hermitian operator
            normalize!(psi)
        end
        # break if truncation error exceeds etol
        if ! isnothing(etol) && truncerr > etol
            break
        end
    end
    return psi, truncerr
end

function mps_evolve(psi0::MPS, ttotal::Int, prob::Tp, eta::Real, obs::AbstractObserver; 
    cutoff::Real=1e-14, maxdim::Int=2<<length(psi0), etol=nothing) where Tp<:Real
    """
    Evolve the MPS `psi0` for `ttotal` time steps with each time step a random unitary operator applied to pairs of sites,
    and a non-Hermitian operator applied to each site with probability `prob` and parameter `eta`.
    """
    psi = copy(psi0)
    T = promote_itensor_eltype(psi)
    sites = siteinds(psi)
    lsize = length(sites)
    
    truncerr = 0.0
    mps_monitor!(obs, psi, 0, truncerr)
    for t in 1:ttotal
        # Apply random unitary operators to pairs of sites
        for j in (iseven(t) + 1):2:lsize-1
            U = op("RdU", sites[j], sites[j+1]; eltype=T)
            err = apply2!(U, psi, j; cutoff=cutoff, maxdim=maxdim)
            truncerr += err
        end
        # Apply non-Hermitian operator to each site with probability `prob` and parameter `eta`
        for j in 1:lsize
            if rand(Tp) >= prob
                continue
            end
            M = op("NH", sites[j]; eta=eta)
            apply!(M, psi, j)
            # Normalize the MPS after applying the non Hermitian operator
            normalize!(psi)
        end
        mps_monitor!(obs, psi, t, truncerr)
        # break if truncation error exceeds etol
        if ! isnothing(etol) && truncerr > etol
            break
        end
    end
    return psi, truncerr
end

function mps_evolve!(psi::MPS, ttotal::Int, prob::Tp, eta::Real; 
    cutoff::Real=1e-14, maxdim::Int=2<<length(psi), etol=nothing) where Tp<:Real
    """
    Evolve the MPS `psi0` for `ttotal` time steps with each time step a random unitary operator applied to pairs of sites,
    and a non-Hermitian operator applied to each site with probability `prob` and parameter `eta`. (inplace version)
    """
    sites = siteinds(psi)
    lsize = length(sites)
    T = promote_itensor_eltype(psi)
    
    truncerr = 0.0
    for t in 1:ttotal
        # Apply random unitary operators to pairs of sites
        for j in (iseven(t) + 1):2:lsize-1
            U = op("RdU", sites[j], sites[j+1]; eltype=T)
            err = apply2!(U, psi, j; cutoff=cutoff, maxdim=maxdim)
            truncerr += err
        end
        # Apply non-Hermitian operator to each site with probability `prob` and parameter `eta`
        for j in 1:lsize
            if rand(Tp) >= prob
                continue
            end
            M = op("NH", sites[j]; eta=eta)
            apply!(M, psi, j)
            # Normalize the MPS after applying the non Hermitian operator
            normalize!(psi)
        end
        # break if truncation error exceeds etol
        if ! isnothing(etol) && truncerr > etol
            break
        end
    end
    return truncerr
end

function mps_evolve!(psi::MPS, ttotal::Int, prob::Tp, eta::Real, obs::AbstractObserver; 
    cutoff::Real=1e-14, maxdim::Int=2<<length(psi), etol=nothing) where Tp<:Real
    """
    Evolve the MPS `psi0` for `ttotal` time steps with each time step a random unitary operator applied to pairs of sites,
    and a non-Hermitian operator applied to each site with probability `prob` and parameter `eta`. (inplace version)
    """
    sites = siteinds(psi)
    lsize = length(sites)
    T = promote_itensor_eltype(psi)
    
    truncerr = 0.0
    mps_monitor!(obs, psi, 0, truncerr)
    for t in 1:ttotal
        # Apply random unitary operators to pairs of sites
        for j in (iseven(t) + 1):2:lsize-1
            U = op("RdU", sites[j], sites[j+1]; eltype=T)
            err = apply2!(U, psi, j; cutoff=cutoff, maxdim=maxdim)
            truncerr += err
        end
        # Apply non-Hermitian operator to each site with probability `prob` and parameter `eta`
        for j in 1:lsize
            if rand(Tp) >= prob
                continue
            end
            M = op("NH", sites[j]; eta=eta)
            apply!(M, psi, j)
            # Normalize the MPS after applying the non Hermitian operator
            normalize!(psi)
        end
        mps_monitor!(obs, psi, t, truncerr)
        # break if truncation error exceeds etol
        if ! isnothing(etol) && truncerr > etol
            break
        end
    end
    return truncerr
end

function mps_monitor!(obs::EntropyObserver{T}, psi::MPS, t::Int, truncerr::Real) where T<:Real
    push!(obs.entropies, ent_entropy(psi, obs.b, obs.n))
    push!(obs.truncerrs, truncerr)
    push!(obs.maxbonds, maxlinkdim(psi))
end

function mps_monitor!(obs::EntrCorrObserver{T}, psi::MPS, t::Int, truncerr::Real) where T<:Real
    push!(obs.entrs, ent_entropy(psi, obs.b, obs.n))
    push!(obs.corrs, correlation_vec(psi, obs.op, obs.op))
    push!(obs.truncerrs, truncerr)
end

function mps_monitor!(obs::EntropyAverager{T}, psi::MPS, t::Int, truncerr::Real) where T<:Real
    """
    Update the mean and SST of entanglement entropy in `obs`.
    Using Welford's algorithm.
    """
    if t > 2 * obs.len
        entr = ent_entropy(psi, obs.b, obs.n)
        delta = entr - obs.entr_mean
        obs.entr_mean += delta / (t - 2 * obs.len)
        obs.entr_sstd += delta * (entr - obs.entr_mean)
    end
end

function mps_monitor!(obs::EntrCorrAverager{T}, psi::MPS, t::Int, truncerr::Real) where T<:Real
    """
    Update the mean and SST of entanglement entropy and correlation function in `obs`.
    Using Welford's algorithm.
    """
    if t > 2 * obs.len
        entr = ent_entropy(psi, obs.b, obs.n)
        corr = correlation_vec(psi, obs.op, obs.op)

        delta_entr = entr - obs.entr_mean
        delta_corr = corr .- obs.corr_mean
        obs.entr_mean += delta_entr / (t - 2 * obs.len)
        obs.corr_mean .+= delta_corr ./ (t - 2 * obs.len)
        obs.entr_sstd += delta_entr * (entr - obs.entr_mean)
        obs.corr_sstd .+= delta_corr .* (corr .- obs.corr_mean)
    end
end
#=
let 
    L, T = 10, 100
    ss = siteinds("S=1/2", L)
    psi = MPS(ComplexF64, ss, "Up")
    obs = EntrCorrAverager{Float64}(L ÷ 2, L; n=1, op="Sz")
    truncerr = mps_evolve!(psi, 40, 0.5, 0.5, obs; cutoff=eps(Float64))
    println("Entropy: ", sqrt.(obs.corr_sstd ./ (T - 2L)))

end
=#