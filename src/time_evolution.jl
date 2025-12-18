using ITensors, ITensorMPS
using LinearAlgebra, Random
include("entanglement.jl")
include("correlation.jl")

struct CalcResult{T}
    mean_entropy::T
    std_entropy::T
    mean_corrs::Vector{T}
    std_corrs::Vector{T}
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
    M = diagm([1, eta])
    return op(M, s)
end

function ITensors.op(::OpName"WM", ::SiteType"S=1/2", s::Index; x::Real, λ::Real=1.0, Δ::Real=1.0)
    """Create a weak measurement operator for the given site index `s` with parameters `x`, `λ`, and `Δ`."""
    # Assuming `x` is a random variable from a Gaussian distribution
    phiUp = exp(-(x-λ)*(x-λ) / (4*Δ*Δ))
    phiDn = exp(-(x+λ)*(x+λ) / (4*Δ*Δ))
    return op([phiUp 0; 0 phiDn], s)
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
    if ortho_lims(psi) != loc:loc
        orthogonalize!(psi, loc)
    end
    A = noprime(psi[loc] * G1)
    psi[loc] = A
end

function apply2!(G2::ITensor, psi::MPS, j1::Int; cutoff::Real=1e-12)
    """
    Apply two adjacent site gate `G2` to the MPS `psi` at sites `j1` and `j1+1` inplace.
    """
    (j1<=0 || j1>= length(psi)) && error("Wrong starting site for two-site gate application.")
    orthogonalize!(psi, j1)
    j2 = j1 + 1
    A = (psi[j1] * psi[j2]) * G2
    noprime!(A)
    linds = uniqueinds(psi[j1], psi[j2])
    psi[j1], S, psi[j2] = svd(A, linds; cutoff=cutoff)
    psi[j2] *= S
    set_ortho_lims!(psi, j2:j2)
end

function mps_evolve(psi0::MPS, ttotal::Int, prob::Tp, eta::Real; cutoff::Real=1e-12) where Tp<:Real
    """
    Evolve the MPS `psi0` for `ttotal` time steps with each time step a random unitary operator applied to pairs of sites,
    and a non-Hermitian operator applied to each site with probability `prob` and parameter `eta`.
    """
    psi = deepcopy(psi0)
    T = promote_itensor_eltype(psi)
    sites = siteinds(psi)
    lsize = length(sites)

    for t in 1:ttotal
        # Apply random unitary operators to pairs of sites
        for j in (iseven(t) + 1):2:lsize-1
            U = op("RdU", sites[j], sites[j+1]; eltype=T)
            apply2!(U, psi, j; cutoff=cutoff)
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
    end
    return psi
end

function mps_evolve!(psi::MPS, ttotal::Int, prob::Tp, eta::Real; cutoff::Real=1e-12) where Tp<:Real
    """
    Evolve the MPS `psi0` for `ttotal` time steps with each time step a random unitary operator applied to pairs of sites,
    and a non-Hermitian operator applied to each site with probability `prob` and parameter `eta`. (inplace version)
    """
    sites = siteinds(psi)
    lsize = length(sites)
    T = promote_itensor_eltype(psi)

    for t in 1:ttotal
        # Apply random unitary operators to pairs of sites
        for j in (iseven(t) + 1):2:lsize-1
            U = op("RdU", sites[j], sites[j+1]; eltype=T)
            apply2!(U, psi, j; cutoff=cutoff)
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
    end
end

function entropy_evolve(psi0::MPS, ttotal::Int, prob::Tp, eta::Real, b::Int, which_ent::Real=1; 
    cutoff::Real=1e-12) where Tp<:Real
    """
    Same with function `mps_evolve!` but with entanglement entropy biparted at site `b` recorded after each time step.
    """
    psi = deepcopy(psi0)
    sites = siteinds(psi) 
    lsize = length(sites)
    T = promote_itensor_eltype(psi)
    # Initialize the entropy vector. 
    entropies = Vector{real(T)}(undef, ttotal+1)
    entropies[1] = Renyi_entropy(psi, b, which_ent)

    for t in 1:ttotal
        # the layer for random unitary operators
        for j in (iseven(t) + 1):2:lsize-1
            U = op("RdU", sites[j], sites[j+1]; eltype=T)
            apply2!(U, psi, j; cutoff=cutoff)
        end
        # the layer for random non-Hermitian gates
        for j in 1:lsize
            if rand(Tp) >= prob
                continue
            end
            M = op("NH", sites[j]; eta=eta)
            apply!(M, psi, j)
            normalize!(psi)
        end
        # Record the entanglement entropy after each time step
        entropies[t+1] = Renyi_entropy(psi, b, which_ent)
    end
    return psi, entropies
end

function entropy_evolve!(psi::MPS, ttotal::Int, prob::Tp, eta::Real, b::Int, which_ent::Real=1; 
    cutoff::Real=1e-12) where Tp<:Real
    """
    Same with function `mps_evolve!` but with entanglement entropy biparted at site `b` recorded after each time step.
    """
    sites = siteinds(psi) 
    lsize = length(sites)
    T = promote_itensor_eltype(psi)
    # Initialize the entropy vector. 
    entropies = Vector{real(T)}(undef, ttotal+1)
    entropies[1] = Renyi_entropy(psi, b, which_ent)

    for t in 1:ttotal
        # the layer for random unitary operators
        for j in (iseven(t) + 1):2:lsize-1
            U = op("RdU", sites[j], sites[j+1]; eltype=T)
            apply2!(U, psi, j; cutoff=cutoff)
        end
        # the layer for random non-Hermitian gates
        for j in 1:lsize
            if rand(Tp) >= prob
                continue
            end
            M = op("NH", sites[j]; eta=eta)
            apply!(M, psi, j)
            normalize!(psi)
        end
        # Record the entanglement entropy after each time step
        entropies[t+1] = Renyi_entropy(psi, b, which_ent)
    end
    return entropies
end

function entropy_avg!(psi::MPS, ttotal::Int, prob::Tp, eta::Real, b::Int, which_ent::Real=1; 
    cutoff::Real=1e-12) where Tp<:Real
    """
    Same with function `mps_evolve!` but with entanglement entropy biparted at site `b` recorded after each time step.
    """
    sites = siteinds(psi) 
    lsize = length(sites)
    T = promote_itensor_eltype(psi)
    sat = 2lsize + 1
    avg, s2 = 0.0, 0.0

    for t in 1:ttotal
        # the layer for random unitary operators
        for j in (iseven(t) + 1):2:lsize-1
            U = op("RdU", sites[j], sites[j+1]; eltype=T)
            apply2!(U, psi, j; cutoff=cutoff)
        end
        # the layer for random non-Hermitian gates
        for j in 1:lsize
            if rand(Tp) >= prob
                continue
            end
            M = op("NH", sites[j]; eta=eta)
            apply!(M, psi, j)
            normalize!(psi)
        end
        if t == sat  #Welford algorithm
            avg = Renyi_entropy(psi, b, which_ent)
        elseif t > sat
            entropy = Renyi_entropy(psi, b, which_ent)
            delta = entropy - avg
            avg += delta /(t + 1 - sat) 
            s2 += delta * (entropy - avg)
        end
    end
    return avg, sqrt(s2 / (ttotal - sat))
end

function entr_corr_evolve(psi0::MPS, ttotal::Int, prob::Tp, eta::Real, b::Int, which_ent::Real=1; 
    cutoff::Real=1e-12) where Tp<:Real
    """
    Same with function `mps_evolve` but with entanglement entropy and correlation function recorded after each time step.
    """
    psi = deepcopy(psi0)
    sites = siteinds(psi) 
    lsize = length(sites)
    T = promote_itensor_eltype(psi)
    # Initialize the entropy vector. 
    entropies = Vector{real(T)}(undef, ttotal+1)
    entropies[1] = Renyi_entropy(psi, b, which_ent)
    corrs = Matrix{real(T)}(undef, lsize, ttotal+1)
    corrs[:, 1] .= correlation_vec(psi, which_op, which_op)

    for t in 1:ttotal
        # the layer for random unitary operators
        for j in (iseven(t) + 1):2:lsize-1
            U = op("RdU", sites[j], sites[j+1]; eltype=T)
            apply2!(U, psi, j; cutoff=cutoff)
        end
        # the layer for random non-Hermitian gates
        for j in 1:lsize
            if rand(Tp) >= prob
                continue
            end
            M = op("NH", sites[j]; eta=eta)
            apply!(M, psi, j)
            normalize!(psi)
        end
        # Record the entanglement entropy and correlation function after each time step
        entropies[t+1] = Renyi_entropy(psi, b, which_ent)
        corrs[:, t+1] .= correlation_vec(psi, which_op, which_op)
    end
    return psi, entropies, corrs
end

function entr_corr_evolve!(psi::MPS, ttotal::Int, prob::Tp, eta::Real, b::Int, which_ent::Real=1, which_op::String="Sz"; 
    cutoff::Real=1e-12) where Tp<:Real
     """
    Same with function `mps_evolve!` but with entanglement entropy biparted at site `b` and correlation vector recorded after each time step.
    """
    sites = siteinds(psi) 
    lsize = length(sites)
    T = promote_itensor_eltype(psi)
    # Initialize the entropy vector. 
    entropies = Vector{real(T)}(undef, ttotal+1)
    entropies[1] = Renyi_entropy(psi, b, which_ent)
    corrs = Matrix{real(T)}(undef, lsize, ttotal+1)
    corrs[:, 1] .= correlation_vec(psi, which_op, which_op)

    for t in 1:ttotal
        # the layer for random unitary operators
        for j in (iseven(t) + 1):2:lsize-1
            U = op("RdU", sites[j], sites[j+1]; eltype=T)
            apply2!(U, psi, j; cutoff=cutoff)
        end
        # the layer for random non-Hermitian gates
        for j in 1:lsize
            if rand(Tp) >= prob
                continue
            end
            M = op("NH", sites[j]; eta=eta)
            apply!(M, psi, j)
            normalize!(psi)
        end
        # Record the entanglement entropy and correlation function after each time step
        entropies[t+1] = Renyi_entropy(psi, b, which_ent)
        corrs[:, t+1] .= correlation_vec(psi, which_op, which_op)
    end
    return entropies, corrs
end

function entr_corr_avg!(psi::MPS, ttotal::Int, prob::Tp, eta::Real, b::Int, which_ent::Real=1, which_op::String="Sz"; 
    cutoff::Real=1e-12) where Tp<:Real
    sites = siteinds(psi) 
    lsize = length(sites)
    T = promote_itensor_eltype(psi)
    restype = real(T)
    sat = 2lsize + 1
    # Initialize the entropy and correlation
    avg_entr, s2_entr = 0.0, 0.0
    avg_corr = zeros(restype, lsize)
    s2_corr = zeros(restype, lsize)

    for t in 1:ttotal
        # the layer for random unitary operators
        for j in (iseven(t) + 1):2:lsize-1
            U = op("RdU", sites[j], sites[j+1]; eltype=T)
            apply2!(U, psi, j; cutoff=cutoff)
        end
        # the layer for random non-Hermitian gates
        for j in 1:lsize
            if rand(Tp) >= prob
                continue
            end
            M = op("NH", sites[j]; eta=eta)
            apply!(M, psi, j)
            normalize!(psi)
        end
        if t == sat
            avg_entr = Renyi_entropy(psi, b, which_ent)
            avg_corr .= correlation_vec(psi, which_op, which_op)
        elseif t > sat
            entr = Renyi_entropy(psi, b, which_ent)
            corr = correlation_vec(psi, which_op, which_op)

            delta_entr = entr - avg_entr
            delta_corr = corr .- avg_corr
            avg_entr += delta_entr / (t + 1 - sat)
            avg_corr .+= delta_corr ./ (t + 1 - sat)
            s2_entr += delta_entr * (entr - avg_entr)
            s2_corr .+= delta_corr .* (corr .- avg_corr)
        end
    end
    std_entr = sqrt(s2_entr / (ttotal - sat))
    std_corr = sqrt.(s2_corr ./ (ttotal - sat))
    return CalcResult{restype}(avg_entr, std_entr, avg_corr, std_corr)
end
#=
let
    L = 10
    p, η = 0.9, 0.0
    ss = siteinds("S=1/2", L)
    psi = MPS(ComplexF64, ss, "Up")

    mps_evolve!(psi, 4L, p, η; cutoff=1e-14)
    
    println(norm(psi))
end
=#
