function expected(psi::MPS, opstr::String, j::Int)
    """Compute the expectation value ⟨ opstr_loc ⟩ for MPS psi."""
    orthogonalize!(psi, j) # proper canonize the MPS s.t left environment is identity
    opm = op(opstr, siteind(psi, j))
    C = psi[j] * opm
    noprime!(C)
    C *= dag(psi[j])
    return real(scalar(C))
end

function correlation(psi::MPS, ops1::String, ops2::String, j1::Int, j2::Int; ortho::Bool=false)
    """Compute the correlation function ⟨ ops1_i ops2_j ⟩ for MPS psi."""
    
    (min(j1, j2) < 1 || max(j1, j2) > length(psi)) && error("The sites do not exist!")
    # orthogonalize the MPS if ortho is false: not orthogonalized
    ortho == false && orthogonalize!(psi, j1)
    op1 = op(ops1, siteind(psi, j1))
    op2 = op(ops2, siteind(psi, j2))
    # Only contract tensors between left and right operators (inclusive)
    C = psi[j1] * op1
    
    if j1 == j2
        Cdag = dag(psi[j1] * op2)
        C *= Cdag
        return real(scalar(C))
    elseif j1 < j2
        noprime!(C)
        ir = linkind(psi, j1)
        C *= dag(prime(psi[j1], ir))

        for n in (j1 + 1) : (j2 - 1)
            C *= psi[n]
            C *= dag(prime(psi[n], tags="Link"))
        end

        C *= noprime(psi[j2] * op2)
        il = linkind(psi, j2 - 1)
        C *= dag(prime(psi[j2], il))

        return real(scalar(C))
    else
        noprime!(C)
        il = linkind(psi, j1 - 1)
        C *= dag(prime(psi[j1], il))

        for n in (j1 - 1) : -1 : (j2 + 1)
            C *= psi[n]
            C *= dag(prime(psi[n], tags="Link"))
        end

        C *= noprime(psi[j2] * op2)
        ir = linkind(psi, j2)
        C *= dag(prime(psi[j2], ir))

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

