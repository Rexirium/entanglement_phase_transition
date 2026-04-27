function expected(psi::MPS, opstr::String, loc::Int)
    """Compute the expectation value ⟨ opstr_loc ⟩ for MPS psi."""
    opm = op(opstr, siteind(psi, loc))
    orthogonalize!(psi, loc) # proper canonize the MPS s.t left environment is identity
    C = psi[loc] * opm
    noprime!(C)
    C *= dag(psi[loc])
    return real(scalar(C))
end

function correlation(psi::MPS, ops1::String, ops2::String, loc1::Int, loc2::Int; ortho::Bool=false)
    """Compute the correlation function ⟨ ops1_i ops2_j ⟩ for MPS psi."""
    
    op1 = op(ops1, siteind(psi, loc1))
    op2 = op(ops2, siteind(psi, loc2))
    (min(loc1, loc2) ≤ 0 || max(loc1, loc2) > length(psi)) && error("The sites do not exist!")
    # orthogonalize the MPS if ortho is false: not orthogonalized
    ortho == false && orthogonalize!(psi, loc1)
    # Only contract tensors between left and right operators (inclusive)
    C = psi[loc1] * op1
    
    if loc1 == loc2
        Cdag = dag(psi[loc1] * op2)
        C *= Cdag
        return real(scalar(C))
    elseif loc1 < loc2
        noprime!(C)
        ir = linkind(psi, loc1)
        C *= dag(prime(psi[loc1], ir))

        for n in (loc1 + 1) : (loc2 - 1)
            C *= psi[n]
            C *= dag(prime(psi[n], tags="Link"))
        end

        C *= noprime(psi[loc2] * op2)
        il = linkind(psi, loc2 - 1)
        C *= dag(prime(psi[loc2], il))

        return real(scalar(C))
    else
        noprime!(C)
        il = linkind(psi, loc1 - 1)
        C *= dag(prime(psi[loc1], il))

        for n in (loc1 - 1) : -1 : (loc2 + 1)
            C *= psi[n]
            C *= dag(prime(psi[n], tags="Link"))
        end

        C *= noprime(psi[loc2] * op2)
        ir = linkind(psi, loc2)
        C *= dag(prime(psi[loc2], ir))

        return real(scalar(C))
    end
end

function correlation_site(psi::MPS, ops1::String, ops2::String)
    """
    Compute the correlation function ⟨ ops1_{L/2} ops2_{j} ⟩ for j = 1,2,...,L
    """
    lsize = length(psi)
    center = lsize ÷ 2
    orthogonalize!(psi, center)

    corrs = zeros(real(promote_itensor_eltype(psi)), lsize)
    for j in 1:lsize
        corrs[j] = correlation(psi, ops1, ops2, center, j; ortho=true)
    end
    return corrs
end

function correlation_dist(psi::MPS, ops1::String, ops2::String, dist::Int)
    """
    Compute the correlation function ⟨ ops1_i ops2_{i+dist} ⟩ for 
    (i, i+dist) symmetric w.r.t the center of MPS .
    """
    lsize = length(psi)
    dist > lsize - 1 && error("The distance is too large!")
    # find symmetric sites
    left = 1 + (lsize - dist - 1) ÷ 2
    right = left + dist
    return correlation(psi, ops1, ops2, left, right)
end

function correlation_dist(psi::MPS, ops1::String, ops2::String)
    """
    Compute the correlation function ⟨ ops1_i ops2_{i+dist} ⟩ for 
    all distances dist = 0, 1, ..., lsize-1, where (i, i+dist) are symmetric
    w.r.t the center of MPS .
    """
    lsize = length(psi)
    corrs = zeros(real(promote_itensor_eltype(psi)), lsize)
    for dist in 0:(lsize - 1)
        # find symmetric sites
        left = 1 + (lsize - dist - 1) ÷ 2
        right = left + dist
        # compute the correlation function for distance dist
        @inbounds corrs[dist+1] = correlation(psi, ops1, ops2, left, right)
    end
    return corrs
end

