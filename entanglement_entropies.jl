function von_Neumann_entropy(psi::MPS, b::Int; cutoff=1e-12)
    """
    Calculate the von Neumann entropy of the MPS `psi` biparted after site `b`.
    """
    (b <= 0 || b >= length(psi)) && return 0.0
    # SVD decomposition to obtain the Schmidt coefficients
    orthogonalize!(psi, b)
    linds =  b==1 ? siteind(psi ,1) : (linkind(psi, b-1), siteind(psi, b))
    _ , S, _ = ITensors.svd(psi[b], linds; cutoff=cutoff)

    ps = diag(S) .* diag(S)
    return - sum(ps .* log2.(ps))
end

function zeroth_entropy(psi::MPS, b::Int; cutoff=1e-12)
    """
    Calculate the zeroth order Renyi entropy of the MPS `psi` biparted after site `b`.
    """
    (b <= 0 || b >= length(psi)) && return 0.0
    # SVD decomposition to obtain the Schmidt coefficients
    orthogonalize!(psi, b)
    linds =  b==1 ? siteind(psi ,1) : (linkind(psi, b-1), siteind(psi, b))
    _ , S, _ = ITensors.svd(psi[b], linds; cutoff=cutoff)
    
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
    # SVD decomposition to obtain the Schmidt coefficients
    orthogonalize!(psi, b)
    linds =  b==1 ? siteind(psi ,1) : (linkind(psi, b-1), siteind(psi, b))
    _ , S, _ = ITensors.svd(psi[b], linds; cutoff=cutoff)

    ps = diag(S) .* diag(S)
    trace = sum(ps .^ n)
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
    Calculate the reduced density matrix eigen values of a region of a single sites `x` from other sites.
    """
    (x ≤ 0 && x > length(psi)) && error("The site does not exist!")
    # obtain the reduced density matrix 
    orthogonalize!(psi, x)
    Ap = prime(dag(psi[x]), tags="Site")
    rho = contract(Ap, psi[x])
    # diagonalize the reduced density matrix
    D, _ = eigen(rho; ishermitian=true, cutoff=cutoff)
    return diag(D)
end

function reduced_density_eigen(psi::MPS, xs::Vector{<:Int}; cutoff=1e-12)
    """
    Calculate the reduced density matrix eigen values of multiple sites `xs` from other sites.
    """
    length(xs) == 0 && return 0.0
    length(xs) == 1 && return reduced_density_eigen(psi, xs[1]; cutoff=cutoff)

    xs = sort(xs)
    a, b = xs[1], xs[end]
    (a ≤ 0 || b > length(psi)) && error("The sites do not exist!")
    # obtain the reduced density matrix
    orthogonalize!(psi, a)
    bra = prime(dag(psi), linkinds(psi)..., siteinds(psi)[xs]...)
    start = a==1 ? psi[a] : prime(psi[a], linkinds(psi, a-1))
    rho = contract(start, bra[a])

    for j in (a+1):(b-1)
        rho *= psi[j]
        rho *= bra[j]
    end
    rho *= prime(psi[b], linkind(psi, b))
    rho *= bra[b]
    # diagonalize the reduced density matrix
    D, _ = eigen(rho; ishermitian=true, cutoff=cutoff)
    return diag(D)
end

function mutual_information_region(psi::MPS, as, bs, n::Real=1; cutoff=1e-12)
    """
    Calculate the mutual information of two separate region of sites `as` and `bs`.
    """
    Sa = Renyi_entropy_region(psi, as, n; cutoff=cutoff)
    Sb = Renyi_entropy_region(psi, bs, n; cutoff=cutoff)
    Sab = Renyi_entropy_region(psi, union(as, bs), n; cutoff=cutoff)
    return Sa + Sb - Sab
end