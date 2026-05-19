function timecorrelation!(psi::MPS, ttotal::Int, tstart::Int, mnt::AbstractMonitor, ops::Tuple; 
    cutoff::Real=1e-14, maxdim::Int=1<<(length(psi) ÷ 2), etol=nothing)
    """
    Evolve the MPS `psi0` for `ttotal` time steps with each time step a random unitary operator applied to pairs of sites,
    and a disentangler `mnt` applied to each site, with properties. (inplace version)
    """
    sites = siteinds(psi)
    lsize = length(sites)
    T = promote_itensor_eltype(psi)

    j1, j2 = ops[2], ops[4]
    op1 = op(ops[1], sites[j1])
    op2 = op(ops[3], sites[j2])
    
    truncerr = zero(real(T))
    timecorrs = zeros(real(T), ttotal - tstart + 1)

    @inbounds for t in 1 : tstart
        # Apply random unitary operators to pairs of sites
        for j in (iseven(t) + 1):2:lsize-1
            U = op("RdU", sites[j], sites[j+1]; eltype=T)
            err = apply2!(U, psi, j; cutoff=cutoff, maxdim=maxdim)
            truncerr += err
        end
        # Apply monitor layer to disentangle the MPS
        truncerr += monitor!(psi, mnt)
       
        # break if truncation error exceeds etol
        if !isnothing(etol) && truncerr > etol
            return timecorrs, truncerr
        end
    end

    phi = copy(psi)
    applyn!(op2, phi, j2)
    # Compute the time correlation function ⟨ ops1_i(t) ops2_j(0) ⟩
    orthogonalize!(psi, j1)
    applyn!(op1, phi, j1)
    timecorrs[1] = real(inner(psi, phi))
    phi[j1] = noprime(phi[j1] * op1) # restore phi to the state
    
    @inbounds for t in tstart + 1 : ttotal
        # Apply random unitary operators to pairs of sites
        for j in (iseven(t) + 1):2:lsize-1
            U = op("RdU", sites[j], sites[j+1]; eltype=T)
            err = apply2!(U, psi, j; cutoff=cutoff, maxdim=maxdim)
            apply2!(U, phi, j; cutoff=cutoff, maxdim=maxdim)
            truncerr += err
        end
        # Apply monitor layer to disentangle the MPS
        truncerr += monitor!(psi, phi, mnt)
        # Compute the time correlation function ⟨ ops1_i(t) ops2_j(0) ⟩
        orthogonalize!(psi, j1)
        applyn!(op1, phi, j1)
        timecorrs[t - tstart + 1] = real(inner(psi, phi))
        phi[j1] *= op1
        noprime!(phi[j1]) # restore phi to the state
        
        if !isnothing(etol) && truncerr > etol
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
    
    j1, j2 = ops[2], ops[4]
    op1 = op(ops[1], sites[j1])
    op2 = op(ops[3], sites[j2])
    
    truncerr = zero(real(T))
    timecorrs = zeros(real(T), ttotal - tstart + 1)

    mps_record!(obs, psi, 0, truncerr)
    @inbounds for t in 1 : tstart
        # Apply random unitary operators to pairs of sites
        for j in (iseven(t) + 1):2:lsize-1
            U = op("RdU", sites[j], sites[j+1]; eltype=T)
            err = apply2!(U, psi, j; cutoff=cutoff, maxdim=maxdim)
            truncerr += err
        end
        # Apply monitor layer to disentangle the MPS
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
    applyn!(op2, phi, j2)
    # Compute the time correlation function ⟨ op1_i(t) op2_j(0) ⟩
    orthogonalize!(psi, j1)
    applyn!(op1, phi, j1)
    timecorrs[1] = real(inner(psi, phi))
    phi[j1] = noprime(phi[j1] * op1) # restore phi to the state
    
    @inbounds for t in tstart + 1 : ttotal
        # Apply random unitary operators to pairs of sites
        for j in (iseven(t) + 1):2:lsize-1
            U = op("RdU", sites[j], sites[j+1]; eltype=T)
            err = apply2!(U, psi, j; cutoff=cutoff, maxdim=maxdim)
            apply2!(U, phi, j; cutoff=cutoff, maxdim=maxdim)
            truncerr += err
        end
        # Apply monitor layer to disentangle the MPS
        truncerr += monitor!(psi, phi, mnt)
        # Monitor the MPS and truncation error
        mps_record!(obs, psi, t, truncerr)
        # Compute the time correlation function ⟨ op1_i(t) op2_j(0) ⟩
        orthogonalize!(psi, j1)
        applyn!(op1, phi, j1)
        timecorrs[t - tstart + 1] = real(inner(psi, phi))
        phi[j1] *= op1
        noprime!(phi[j1]) # restore phi to the state
        
        if !isnothing(etol) && truncerr > etol
            obs.accept = false
            return timecorrs, truncerr
        end
    end
    return timecorrs, truncerr
end
