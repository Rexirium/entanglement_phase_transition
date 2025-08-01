function von_Neumann_entropy(psi::MPS, b::Int; cutoff=1e-12)
    """
    Calculate the von Neumann entropy of the MPS `psi` bipartited before site `b`.
    """
    b <= 0 && return 0.0
    psi_tmp = orthogonalize(psi,b)
    llink = linkinds(psi_tmp,b-1)
    lsite = siteinds(psi_tmp,b)
    U,S,V = svd(psi_tmp[b], (llink..., lsite...); cutoff=cutoff)
    SvN = 0.0
    for k = 1:dim(S,1)
        p = S[k, k]*S[k, k]
        SvN -= p*log2(p)
    end
    return SvN
end

function zeroth_entropy(psi::MPS, b::Int; cutoff=1e-12)
    """
    Calculate the zeroth order Renyi entropy of the MPS `psi` bipartited before site `b`.
    """
    b <= 0 && return 0.0
    psi_tmp = orthogonalize(psi,b)
    llink = linkinds(psi_tmp,b-1)
    lsite = siteinds(psi_tmp,b)
    U,S,V = svd(psi_tmp[b], (llink..., lsite...); cutoff=cutoff)
    chi = dim(S,1)
    return log2(chi)
end

function Renyi_entropy(psi::MPS, b::Int, n::Real; cutoff=1e-12)
    """
    Calculate the n-th order Renyi entropy of the MPS `psi` bipartited before site `b`.
    """
    b <= 0 && return 0.0
    n == 0 && return zeroth_entropy(psi, b; cutoff=cutoff)
    n == 1 && return von_Neumann_entropy(psi, b; cutoff=cutoff)
    psi_tmp = orthogonalize(psi,b)
    llink = linkinds(psi_tmp,b-1)
    lsite = siteinds(psi_tmp,b)
    U,S,V = svd(psi_tmp[b], (llink..., lsite...); cutoff=cutoff)
    svals = diag(S)
    trace = sum((svals .* svals).^n)
    return log2(trace)/(1-n)
end
