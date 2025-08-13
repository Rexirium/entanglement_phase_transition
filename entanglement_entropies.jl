function von_Neumann_entropy(psi::MPS, b::Int; cutoff=1e-12)
    """
    Calculate the von Neumann entropy of the MPS `psi` biparted after site `b`.
    """
    (b <= 0 || b >= length(psi)) && return 0.0
    psi_tmp = orthogonalize(psi, b)
    llink = linkinds(psi_tmp, b-1)
    lsite = siteinds(psi_tmp, b)
    _ , S, _ = ITensors.svd(psi_tmp[b], (llink..., lsite...); cutoff=cutoff)
    SvN = 0.0
    for k = 1:dim(S,1)
        p = S[k, k] * S[k, k]
        SvN -= p * log2(p)
    end
    return SvN
end

function zeroth_entropy(psi::MPS, b::Int; cutoff=1e-12)
    """
    Calculate the zeroth order Renyi entropy of the MPS `psi` biparted after site `b`.
    """
    (b <= 0 || b >= length(psi)) && return 0.0
    psi_tmp = orthogonalize(psi, b)
    llink = linkinds(psi_tmp, b-1)
    lsite = siteinds(psi_tmp, b)
    _ , S, _ = ITensors.svd(psi_tmp[b], (llink..., lsite...); cutoff=cutoff)
    chi = dim(S,1)
    return log2(chi)
end

function Renyi_entropy(psi::MPS, b::Int, n::Real; cutoff=1e-12)
    """
    Calculate the n-th order Renyi entropy of the MPS `psi` biparted after site `b`.
    """
    (b <= 0 || b >= length(psi)) && return 0.0
    n == 0 && return zeroth_entropy(psi, b; cutoff=cutoff)
    n == 1 && return von_Neumann_entropy(psi, b; cutoff=cutoff)
    psi_tmp = orthogonalize(psi, b)
    llink = linkinds(psi_tmp, b-1)
    lsite = siteinds(psi_tmp, b)
    _ , S, _ = ITensors.svd(psi_tmp[b], (llink..., lsite...); cutoff=cutoff)
    svals = diag(S)
    trace = sum((svals .* svals).^n)
    return log2(trace)/(1-n)
end

function von_Neumann_entropy_single(psi::MPS, x::Int; cutoff = 1e-12)
    """
    Calculate the von Neumann entropy of a single site `x` from other sites.
    """
    (x < 0 && x > length(psi)) && error("The site does not exist!")
    psi_tmp = orthogonalize(psi, x)
    Ap = prime(psi_tmp[x], tags="Site")
    Ac = conj(psi_tmp[x])
    M = contract(Ap, Ac)
    D, _ = eigen(M; ishermitian=true)
    ps = diag(D)
    ps = ps[ps .> cutoff]
    return -sum(ps .* log2.(ps))
end

function zeroth_entropy_single(psi::MPS, x::Int; cutoff = 1e-12)
    """
    Calculate the zeroth order Renyi entropy of a single site `x` from other sites.
    """
    (x < 0 && x > length(psi)) && error("The site does not exist!")
    psi_tmp = orthogonalize(psi, x)
    Ap = prime(psi_tmp[x], tags="Site")
    Ac = conj(psi_tmp[x])
    M = contract(Ap, Ac)
    D, _ = eigen(M; ishermitian=true)
    ps = diag(D)
    ps = ps[ps .> cutoff]
    return log2(length(ps))
end

function Renyi_entropy_single(psi::MPS, x::Int, n::Real; cutoff = 1e-12)
    """
    Calculate the n-th order Renyi entropy of a single site `x` from other sites.
    """
    (x < 0 && x > length(psi)) && error("The site does not exist!")
    n == 0 && return zeroth_entropy_single(psi, x; cutoff=cutoff)
    n == 1 && return von_Neumann_entropy_single(psi, x; cutoff=cutoff)
    psi_tmp = orthogonalize(psi, x)
    Ap = prime(psi_tmp[x], tags="Site")
    Ac = conj(psi_tmp[x])
    M = contract(Ap, Ac)
    D, _ = eigen(M; ishermitian=true)
    ps = diag(D)
    ps = ps[ps .> cutoff]
    trace = sum(ps .^ n)
    return log2(trace)/(1-n)
end

function mutual_information(psi::MPS, a::Int, b::Int, n::Real; cutoff=1e-12)
    """
    Calculate the mutual information of two separate spin at site `a` and `b`
    """
end