function correlation(psi::MPS, ops1::String, ops2::String, i::Int, j::Int)
    """Compute the correlation function ⟨ ops1_i ops2_j ⟩ for MPS psi."""
    left, right = sort((i, j))
    idxs = siteinds(psi)[[left, right]]
    op1 = op(ops1, idxs[1])
    op2 = op(ops2, idxs[2])
    (left ≤ 0 || right > length(psi)) && error("The sites do not exist!")
    orthogonalize!(psi, left) # proper canonize the MPS s.t left environment is identity
    # Only contract tensors between left and right operators (inclusive)
    C = psi[left] * op1
    if left == right
        Cdag = dag(psi[left] * op2)
        C *= Cdag
        return real(scalar(C))
    end
    ir = linkind(psi, left)
    C *= dag(prime(prime(psi[left], tags="Site"), ir))
    for n in (left+1):(right-1)
        C *= psi[n]
        C *= dag(prime(psi[n], tags="Link"))
    end
    C *= psi[right] * op2
    il = linkind(psi, right-1)
    C *= dag(prime(prime(psi[right], tags="Site"), il))
    return real(scalar(C))
end

function correlation(psi::MPS, ops1::String, ops2::String, dist::Int)
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

function correlation_vec(psi::MPS, ops1::String, ops2::String)
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
        corrs[dist+1] = correlation(psi, ops1, ops2, left, right)
    end
    return corrs
end
#=
let 
    ss = siteinds("S=1/2", 10)
    psi = randomMPS(ss; linkdims=1)
    orthogonalize!(psi, 5)
    @time println(correlation_vec(psi, "Sz", "Sz"))
end
=#

