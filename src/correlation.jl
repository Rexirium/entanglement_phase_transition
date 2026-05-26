function ITensorMPS.expect(psi::MPS, opstr::String, j::Int)
    """Compute the expectation value ⟨ opstr_loc ⟩ for MPS psi."""
    orthogonalize!(psi, j) # proper canonize the MPS s.t left environment is identity
    opm = op(opstr, siteind(psi, j))
    C = psi[j] * opm
    noprime!(C)
    C *= dag(psi[j])
    return real(scalar(C))
end

function ITensorMPS.expect(psi::InfMPS, opstr::String, j::Int)
    j0 = mod1(j - 1, psi.len_uc)
    opm = op(opstr, siteind(psi, j))
    C = prime(psi.Lambdas[j0], "RLink") * psi.Gammas[j] * prime(psi.Lambdas[j], "LLink")
    C *= opm
end

function correlation(psi::MPS, ops1::String, ops2::String, j1::Int, j2::Int; ortho::Bool=true)
    """Compute the correlation function ⟨ ops1_i ops2_j ⟩ for MPS psi."""

    ja, jb = minmax(j1, j2)
    (ja < 1 || jb > length(psi)) && error("The sites do not exist!")
    # orthogonalize the MPS if ortho is false: not orthogonalized
    ortho == false && orthogonalize!(psi, j1)

    opsa, opsb = j1 <= j2 ? (ops1, ops2) : (ops2, ops1)
    opa = op(opsa, siteind(psi, ja))
    opb = op(opsb, siteind(psi, jb))

    C = psi[ja] * opa

    if ja == jb
        Cdag = dag(psi[ja] * opb)
        C *= Cdag
        return real(scalar(C))
    end
    # Only contract tensors between left and right operators (inclusive)
    noprime!(C)
    C *= dag(prime(psi[ja], tags="Link,l=$ja"))

    for n in (ja + 1) : (jb - 1)
        C *= psi[n]
        C *= dag(prime(psi[n], tags="Link"))
    end

    C *= noprime(psi[jb] * opb)
    C *= dag(prime(psi[jb], tags="Link,l=$(jb-1)"))

    return real(scalar(C))
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

