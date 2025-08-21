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

function von_Neumann_entropy_region(psi::MPS, xs; cutoff = 1e-12)
    """
    Calculate the von Neumann entropy of a single site `x` from other sites.
    """
    ps = reduced_density_eigen(psi, xs; cutoff=cutoff)
    return - sum(ps .* log2.(ps))
end

function zeroth_entropy_region(psi::MPS, xs; cutoff = 1e-12)
    """
    Calculate the zeroth order Renyi entropy of a region of sites `x` from other sites.
    """
    ps = reduced_density_eigen(psi, xs; cutoff=cutoff)
    return log2(length(ps))
end

function Renyi_entropy_region(psi::MPS, xs, n::Real; cutoff = 1e-12)
    """
    Calculate the n-th order Renyi entropy of a region of sites `x` from other sites.
    """
    n == 0 && return zeroth_entropy_region(psi, xs; cutoff=cutoff)
    n == 1 && return von_Neumann_entropy_region(psi, xs; cutoff=cutoff)
    ps = reduced_density_eigen(psi, xs; cutoff=cutoff)
    trace = sum(ps .^ n)
    return log2(trace)/(1-n)
end

function reduced_density_eigen(psi::MPS, x::Int; cutoff=1e-12)
    """
    Calculate the reduced density matrix eigen values of a region of sites `x` from other sites.
    """
    (x < 0 && x > length(psi)) && error("The site does not exist!")
    psi_tmp = orthogonalize(psi, x)
    Ap = prime(dag(psi_tmp[x]), tags="Site")
    rho = contract(Ap, psi_tmp[x])
    D, _ = eigen(rho; ishermitian=true)
    ps = diag(D)
    return ps[ps .> cutoff]
end

function reduced_density_eigen(psi::MPS, xs::Vector{<:Int}; cutoff=1e-12)
    """
    Calculate the reduced density matrix eigen values of multiple sites `xs` from other sites.
    """
    length(xs) == 0 && error("No sites provided!")
    length(xs) == 1 && return reduced_density_eigen(psi, xs[1]; cutoff=cutoff)

    xs = sort(xs)
    a, b = xs[1], xs[end]
    (a < 0 || b > length(psi)) && error("The sites do not exist!")
    ket = orthogonalize(psi, a)
    bra = prime(dag(ket), linkinds(ket)..., siteinds(ket)[xs]...)
    start = prime(ket[a], linkinds(ket, a-1))
    rho = contract(start, bra[a])

    for j in (a+1):(b-1)
        rho *= ket[j]
        rho *= bra[j]
    end
    rho *= prime(ket[b], linkinds(ket, b))
    rho *= bra[b]

    D, _ = eigen(rho; ishermitian=true)
    ps = diag(D)
    return ps[ps .> cutoff]
end

function mutual_information_region(psi::MPS, as, bs, n::Real; cutoff=1e-12)
    """
    Calculate the mutual information of two separate region of sites `as` and `bs`.
    """
    Sa = Renyi_entropy_region(psi, as, n; cutoff=cutoff)
    Sb = Renyi_entropy_region(psi, bs, n; cutoff=cutoff)
    Sab = Renyi_entropy_region(psi, union(as, bs), n; cutoff=cutoff)
    return Sa + Sb - Sab
end