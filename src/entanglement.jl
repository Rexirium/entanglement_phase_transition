function von_Neumann_entropy(psi::MPS, b::Int)
    """
    Calculate the von Neumann entropy of the MPS `psi` biparted after site `b`.
    """
    (b <= 0 || b >= length(psi)) && return 0.0
    # SVD decomposition to obtain the Schmidt coefficients
    orthogonalize!(psi, b)
    linds = uniqueinds(psi[b], psi[b+1])
    _ , S, _ = ITensors.svd(psi[b], linds)

    ps = diag(S) .* diag(S)
    ps = ps[ps.>0] # remove zero probabilities
    return - sum(ps .* log2.(ps))
end

function zeroth_entropy(psi::MPS, b::Int)
    """
    Calculate the zeroth order Renyi entropy of the MPS `psi` biparted after site `b`.
    """
    (b <= 0 || b >= length(psi)) && return 0.0
    # SVD decomposition to obtain the Schmidt coefficients
    orthogonalize!(psi, b)
    return log2(linkdim(psi, b))
end

function Renyi_entropy(psi::MPS, b::Int, n::Real)
    """
    Calculate the n-th order Renyi entropy of the MPS `psi` biparted after site `b`.
    """
    (b <= 0 || b >= length(psi)) && return 0.0
    n == 0 && return zeroth_entropy(psi, b)
    n == 1 && return von_Neumann_entropy(psi, b)
    # SVD decomposition to obtain the Schmidt coefficients
    orthogonalize!(psi, b)
    linds = uniqueinds(psi[b], psi[b+1])
    _ , S, _ = ITensors.svd(psi[b], linds)

    ps = diag(S) .* diag(S)
    ps = ps[ps.>0]
    trace = sum(ps .^ n)
    return log2(trace)/(1-n)
end

function von_Neumann_entropy_region(psi::MPS, xs)
    """
    Calculate the von Neumann entropy of a single site `x` from other sites.
    """
    ps = reduced_density_eigen(psi, xs)
    return - sum(ps .* log2.(ps))
end

function zeroth_entropy_region(psi::MPS, xs)
    """
    Calculate the zeroth order Renyi entropy of a region of sites `x` from other sites.
    """
    ps = reduced_density_eigen(psi, xs)
    return log2(length(ps))
end

function Renyi_entropy_region(psi::MPS, xs, n::Real)
    """
    Calculate the n-th order Renyi entropy of a region of sites `x` from other sites.
    """
    n == 0 && return zeroth_entropy_region(psi, xs)
    n == 1 && return von_Neumann_entropy_region(psi, xs)

    ps = reduced_density_eigen(psi, xs)
    trace = sum(ps .^ n)
    return log2(trace)/(1-n)
end

function reduced_density_eigen(psi::MPS, x::Int)
    """
    Calculate the reduced density matrix eigen values of a region of a single sites `x` from other sites.
    """
    (x ≤ 0 && x > length(psi)) && error("The site does not exist!")
    # obtain the reduced density matrix 
    orthogonalize!(psi, x)
    Ap = prime(dag(psi[x]), tags="Site")
    rho = contract(Ap, psi[x])
    # diagonalize the reduced density matrix
    D, _ = eigen(rho; ishermitian=true)
    ps = diag(D)
    return ps[ps.>0]  # remove zero probabilities
end

function reduced_density_eigen(psi::MPS, xs::Vector{<:Int})
    """
    Calculate the reduced density matrix eigen values of multiple sites `xs` from other sites.
    """
    length(xs) == 0 && return 0.0
    length(xs) == 1 && return reduced_density_eigen(psi, xs[1])

    xs = sort(xs)
    a, b = xs[1], xs[end]
    (a ≤ 0 || b > length(psi)) && error("The sites do not exist!")
    # obtain the reduced density matrix
    orthogonalize!(psi, a)

    rho = psi[a]
    ir = linkind(psi, a)
    rho *= dag(prime(prime(psi[a], tags="Site"), ir))
    for j in (a+1):(b-1)
        rho *= psi[j]
        rho *= dag(prime(psi[j]))
    end
    rho *= psi[b]
    il = linkind(psi, b-1)
    rho *= dag(prime(prime(psi[b], tags="Site"), il))

    # diagonalize the reduced density matrix
    D, _ = eigen(rho; ishermitian=true)
    ps = diag(D)
    return ps[ps.>0]  # remove zero probabilities
end

function mutual_information_region(psi::MPS, as, bs, n::Real=1)
    """
    Calculate the mutual information of two separate region of sites `as` and `bs`.
    """
    Sa = Renyi_entropy_region(psi, as, n)
    Sb = Renyi_entropy_region(psi, bs, n)
    Sab = Renyi_entropy_region(psi, union(as, bs), n)
    return Sa + Sb - Sab
end