abstract type AbstractDisentangler end

struct NHDisentangler{Tp <: Real} <: AbstractDisentangler
    """
    Store the parameters for the non-Hermitian disentangler. 
    `prob` is the probability of applying the non-Hermitian operator, 
    and `eta` is the parameter of the non-Hermitian operator.
    """
    prob::Tp
    eta::Tp
end

struct NHCNOTDisentangler{Tp <: Real} <: AbstractDisentangler
    """
    Store the parameters for the CNOT-based non-Hermitian disentangler.
    """
    prob::Tp
    eta::Tp
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
            err = apply3!(M, psi, j)
            truncerr += err
            normalize!(psi)
        end
    end
    return truncerr
end

function timeevolve!(psi::MPS, ttotal::Int, dent::AbstractDisentangler; 
    cutoff::Real=1e-14, maxdim::Int=1<<(length(psi) ÷ 2), etol=nothing)
    """
    Evolve the MPS `psi0` for `ttotal` time steps with each time step a random unitary operator 
    applied to pairs of sites, and a disentangler `dent` applied to each site. (inplace version)
    """
    sites = siteinds(psi)
    lsize = length(sites)
    T = promote_itensor_eltype(psi)
    
    truncerr = zero(real(T))
    @inbounds for t in 1:ttotal
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
            return truncerr
        end
    end
    return truncerr
end

function timeevolve!(psi::MPS, ttotal::Int, dent::AbstractDisentangler, obs::AbstractObserver; 
    cutoff::Real=1e-14, maxdim::Int=1<<(length(psi) ÷ 2), etol=nothing)
    """
    Evolve the MPS `psi0` for `ttotal` time steps with each time step a random unitary operator applied to pairs of sites,
    and a disentangler `dent` applied to each site, with properties assigned in `obs` stored for each time step. (inplace version)
    """
    sites = siteinds(psi)
    lsize = length(sites)
    T = promote_itensor_eltype(psi)
    
    truncerr = zero(real(T))
    mps_monitor!(obs, psi, 0, truncerr)
    @inbounds for t in 1:ttotal
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
            return truncerr
        end
    end
    return truncerr
end