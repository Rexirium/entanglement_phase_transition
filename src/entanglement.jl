function ent_entropy(ps::NDTensors.Tensor, n::Real=1)
    """
    Calculate the n-th order Renyi entropy from the eigenvalues ps.
    """
    if n == 0
        return log2(length(ps))
    elseif n == 1
        return -sum(ps .* log2.(ps))
    else
        trace = sum(ps .^ n)
        return log2(trace)/(1-n)
    end
end

function ent_entropy(psi::MPS, b::Int, n::Real=1)
    """
    Calculate the n-th order Renyi entropy of the MPS `psi` biparted after site `b`.
    """
    (b <= 0 || b >= length(psi)) && return 0.0
    # SVD decomposition to obtain the Schmidt coefficients
    orthogonalize!(psi, b)
    linds = uniqueinds(psi[b], psi[b+1])
    _ , S, _ = ITensors.svd(psi[b], linds)

    ps = diag(S) .* diag(S)
    ps = ps[ps.>0] # remove zero probabilities
    return ent_entropy(ps, n)
end

function ent_entropy(psi::MPS, xs::Vector{<:Int}, n::Real=1)
    """
    Calculate the n-th order Renyi entropy of a region of sites `xs` from other sites.
    """
    ps = reduced_density_eigen(psi, xs)
    return ent_entropy(ps, n)
end

function von_neumann_entropy(psi::MPS, b::Int)
    """
    Calculate the von Neumann entropy of the MPS `psi` biparted after site `b`.
    """
    return ent_entropy(psi, b, 1)
end

function zeroth_entropy(psi::MPS, b::Int)
    """
    Calculate the zeroth order Renyi entropy of the MPS `psi` biparted after site `b`.
    """
    (b <= 0 || b >= length(psi)) && return 0.0
    # SVD decomposition to obtain the Schmidt coefficients
    orthogonalize!(psi, b)
    truncate!(psi, b) 
    return log2(linkdim(psi, b))
end

function renyi_entropy(psi::MPS, b::Int, n::Real)
    """
    Calculate the n-th order Renyi entropy of the MPS `psi` biparted after site `b`.
    """
    return ent_entropy(psi, b, n)
end

function von_neumann_entropy_region(psi::MPS, xs::Vector{<:Int})
    """
    Calculate the von Neumann entropy of a region of sites `xs` from other sites.
    """
    return ent_entropy(psi, xs, 1)
end

function zeroth_entropy_region(psi::MPS, xs::Vector{<:Int})
    """
    Calculate the zeroth order Renyi entropy of a region of sites `xs` from other sites.
    """
    return ent_entropy(psi, xs, 0)
end

function renyi_entropy_region(psi::MPS, xs::Vector{<:Int}, n::Real)
    """
    Calculate the n-th order Renyi entropy of a region of sites `xs` from other sites.
    """
    return ent_entropy(psi, xs, n)
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
        if j in xs
            rho *= dag(prime(psi[j]))
        else
            rho *= dag(prime(psi[j], tags="Link"))
        end
    end
    rho *= psi[b]
    il = linkind(psi, b-1)
    rho *= dag(prime(prime(psi[b], tags="Site"), il))

    # diagonalize the reduced density matrix
    D, _ = eigen(rho; ishermitian=true)
    ps = diag(D)
    return ps[ps.>0]  # remove zero probabilities
end

function mutual_information_region(psi::MPS, as::Vector{<:Int}, bs::Vector{<:Int}, n::Real=1)
    """
    Calculate the mutual information of two separate region of sites `as` and `bs`.
    """
    Sa = ent_entropy(psi, as, n)
    Sb = ent_entropy(psi, bs, n)
    Sab = ent_entropy(psi, sort(union(as, bs)), n)
    return Sa + Sb - Sab
end
