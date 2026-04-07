function monitor!(psi::MPS, phi::MPS, mnt::NHMonitor{Tp}) where Tp<:AbstractFloat
    """
    Apply the non-Hermitian disentangler to the MPS `psi` inplace, and apply the same operation to `phi`.
    """
    for j in length(psi):-1:1
        if rand() < mnt.prob
            M = op("NH", siteind(psi, j); eta=mnt.eta)
            apply1!(M, psi, j)
            apply1!(M, phi, j)
            normalize!(psi)
            normalize!(phi)
        end
    end
    return zero(Tp)
end

function monitor!(psi::MPS, phi::MPS, mnt::NHCNOTMonitor{Tp}) where Tp<:AbstractFloat
    """
    Apply the CNOT-based non-Hermitian disentangler to the MPS `psi` inplace.
    """
    ss = siteinds(psi)
    truncerr = zero(Tp)
    for j in (length(psi)-1):-1:2
        if rand() < mnt.prob
            M = op("NHCNOT", ss[j-1 : j+1]...; eta=mnt.eta)
            err = apply3!(M, psi, j)
            apply3!(M, phi, j)
            truncerr += err
            normalize!(psi)
            normalize!(phi)
        end
    end
    return truncerr
end

function monitor!(psi::MPS, phi::MPS, mnt::PMMonitor{Tp}) where Tp<:AbstractFloat
    """
    Apply the projective measurement disentangler to the MPS `psi` inplace, and apply the same operation to `phi`.
    """
    for j in length(psi):-1:1
        if rand() < mnt.probs[j]
            proj_measure!(psi, j)
            proj_measure!(phi, j)
        end
    end
    return zero(Tp)
end

function timecorrelation!(psi::MPS, ttotal::Int, tstart::Int, mnt::AbstractMonitor, ops::Tuple; 
    cutoff::Real=1e-14, maxdim::Int=1<<(length(psi) ÷ 2), etol=nothing)
    """
    Evolve the MPS `psi0` for `ttotal` time steps with each time step a random unitary operator applied to pairs of sites,
    and a disentangler `mnt` applied to each site, with properties. (inplace version)
    """
    sites = siteinds(psi)
    lsize = length(sites)
    T = promote_itensor_eltype(psi)

    op1 = op(ops[1], sites[ops[2]])
    op2 = op(ops[3], sites[ops[4]])
    
    truncerr = zero(real(T))
    timecorrs = zeros(real(T), ttotal - tstart)

    @inbounds for t in 1 : tstart
        # Apply random unitary operators to pairs of sites
        for j in (iseven(t) + 1):2:lsize-1
            U = op("RdU", sites[j], sites[j+1]; eltype=T)
            err = apply2!(U, psi, j; cutoff=cutoff, maxdim=maxdim)
            truncerr += err
        end
        # Apply non-Hermitian disentanglers
        truncerr += monitor!(psi, mnt)
       
        # break if truncation error exceeds etol
        if !isnothing(etol) && truncerr > etol
            obs.accept = false
            return timecorrs, truncerr
        end
    end

    phi = copy(psi)
    apply1!(op2, phi, ops[4])
    
    @inbounds for t in tstart + 1 : ttotal
        # Compute the time correlation function ⟨ ops1_i(t) ops2_j(0) ⟩
        orthogonalize!(psi, ops[2])
        apply1!(op1, phi, ops[2])
        timecorrs[t - tstart] = real(inner(phi, psi))
        apply1!(op1, phi, ops[2]) # restore phi to the state
        # Apply random unitary operators to pairs of sites
        for j in (iseven(t) + 1):2:lsize-1
            U = op("RdU", sites[j], sites[j+1]; eltype=T)
            err = apply2!(U, psi, j; cutoff=cutoff, maxdim=maxdim)
            apply2!(U, phi, j; cutoff=cutoff, maxdim=maxdim)
            truncerr += err
        end
        # Apply non-Hermitian disentanglers
        truncerr += monitor!(psi, phi, mnt)
        
        if !isnothing(etol) && truncerr > etol
            obs.accept = false
            return timecorrs, truncerr
        end
    end
    return timecorrs, truncerr
end

function timecorrelation!(psi::MPS, ttotal::Int, tstart::Int, mnt::AbstractMonitor, ops::Tuple, obs::AbstractObserver; 
    cutoff::Real=1e-14, maxdim::Int=1<<(length(psi) ÷ 2), etol=nothing)
    """
    Evolve the MPS `psi0` for `ttotal` time steps with each time step a random unitary operator applied to pairs of sites,
    and a disentangler `mnt` applied to each site, with properties assigned in `obs` stored for each time step. (inplace version)
    """
    sites = siteinds(psi)
    lsize = length(sites)
    T = promote_itensor_eltype(psi)

    op1 = op(ops[1], sites[ops[2]])
    op2 = op(ops[3], sites[ops[4]])
    
    truncerr = zero(real(T))
    timecorrs = zeros(real(T), ttotal - tstart)

    mps_record!(obs, psi, 0, truncerr)
    @inbounds for t in 1 : tstart
        # Apply random unitary operators to pairs of sites
        for j in (iseven(t) + 1):2:lsize-1
            U = op("RdU", sites[j], sites[j+1]; eltype=T)
            err = apply2!(U, psi, j; cutoff=cutoff, maxdim=maxdim)
            truncerr += err
        end
        # Apply non-Hermitian disentanglers
        truncerr += monitor!(psi, mnt)
        # Monitor the MPS and truncation error
        mps_record!(obs, psi, t, truncerr)
        # break if truncation error exceeds etol
        if !isnothing(etol) && truncerr > etol
            obs.accept = false
            return timecorrs, truncerr
        end
    end

    phi = copy(psi)
    apply1!(op2, phi, ops[4])
    
    @inbounds for t in tstart + 1 : ttotal
        # Compute the time correlation function ⟨ ops1_i(t) ops2_j(0) ⟩
        orthogonalize!(psi, ops[2])
        apply1!(op1, phi, ops[2])
        timecorrs[t - tstart] = real(inner(phi, psi))
        apply1!(op1, phi, ops[2]) # restore phi to the state
        # Apply random unitary operators to pairs of sites
        for j in (iseven(t) + 1):2:lsize-1
            U = op("RdU", sites[j], sites[j+1]; eltype=T)
            err = apply2!(U, psi, j; cutoff=cutoff, maxdim=maxdim)
            apply2!(U, phi, j; cutoff=cutoff, maxdim=maxdim)
            truncerr += err
        end
        # Apply non-Hermitian disentanglers
        truncerr += monitor!(psi, phi, mnt)
        # Monitor the MPS and truncation error
        mps_record!(obs, psi, t, truncerr)
        
        if !isnothing(etol) && truncerr > etol
            obs.accept = false
            return timecorrs, truncerr
        end
    end
    return timecorrs, truncerr
end
