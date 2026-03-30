const CNOT13 = begin
    X = sparse([0 1; 1 0])
    Id = sparse(I, 4, 4)
    blockdiag(Id, X, X)
end

function ITensors.op(::OpName"RdU", ::SiteType"S=1/2", s::Index...; eltype::DataType=ComplexF64)
    """
    Create a random unitary operator for the given site indices `s`.
    """
    M = randn(eltype, 4, 4)
    Q, _ = NDTensors.qr_positive(M)
    return op(Q, s...)
end

function ITensors.op(::OpName"NH", ::SiteType"S=1/2", s::Index; eta::Real)
    """
    Create a non-Hermitian operator for the given site index `s` with parameter `eta`.
    """
    M = diagm(shuffle([one(eta), eta]))
    return op(M, s)
end

function ITensors.op(::OpName"NHCNOT", ::SiteType"S=1/2", s::Index...; eta::Real)
    """
    Create a CNOT-based non-Hermitian operator for the given site indices `s1` and `s2` with parameter `eta`.
    """
    id = ones(typeof(eta), 2)
    dd = kron(id, shuffle([one(eta), eta]), id)
    return op(diagm(dd) * CNOT13, s...)
end

function ITensors.op(::OpName"WM", ::SiteType"S=1/2", s::Index; x::Real, λ::Real=1.0, Δ::Real=1.0)
    """Create a weak measurement operator for the given site index `s` with parameters `x`, `λ`, and `Δ`."""
    # Assuming `x` is a random variable from a Gaussian distribution
    phiUp = exp(-(x-λ)*(x-λ) / (4*Δ*Δ))
    phiDn = exp(-(x+λ)*(x+λ) / (4*Δ*Δ))
    return op([phiUp 0; 0 phiDn], s)
end

function proj_measure!(psi::MPS, loc::Int)
    """Perform a projective measurement on the MPS `psi` at site `loc` with outcome `:Up` or `:Dn`."""
    (loc <= 0 || loc > length(psi)) && return psi
    # Orthogonalize the MPS at site `loc`
    s = siteind(psi, loc)
    projUp = op("ProjUp", s)
    orthogonalize!(psi, loc)
    # Calculate the probability of measuring "Up"
    probUp = real(inner(prime(psi[loc], tags="Site"), projUp, psi[loc]))
    samp = rand()
    if samp < probUp
        apply1!(projUp, psi, loc)
    else
        projDn = op("ProjDn", s)
        apply1!(projDn, psi, loc)
    end
    normalize!(psi)
end

function weak_measure!(psi::MPS, loc::Int, para::Tuple{T, T}=(1.0, 1.0)) where T<:Real
    """Perform a weak measurement on the MPS `psi` at site `loc` with parameters `λ` and `Δ`."""
    (loc <= 0 || loc > length(psi)) && return psi
    # Orthogonalize the MPS at site `loc`
    s = siteind(psi, loc)
    λ, Δ = para
    
    proj = op("ProjUp", s)
    orthogonalize!(psi, loc)
    # Calculate the probability of measuring "Up"
    probUp = real(inner(prime(psi[loc], tags="Site"), proj, psi[loc]))
    samp = rand(T)
    # generate a random variable from a Gaussian distribution
    x = samp < probUp ? λ + Δ*randn(T) : -λ + Δ*randn(T)
    M = op("WM", s; x = x, λ = λ, Δ = Δ)
    # Apply the weak measurement operator
    apply1!(M, psi, loc)
    normalize!(psi)
end

function apply1!(G1::ITensor, psi::MPS, loc::Int)
    """
    Apply the gate `G1` to the MPS `psi` at site `loc` inplace.
    """
    orthogonalize!(psi, loc)
    psi[loc] = noprime(psi[loc] * G1)
end

function apply2!(G2::ITensor, psi::MPS, j1::Int; cutoff::Real=1e-14, maxdim::Int=2*maxlinkdim(psi))
    """
    Apply two adjacent site gate `G2` to the MPS `psi` at sites `j1` and `j1+1` inplace.
    """
    (j1<=0 || j1>= length(psi)) && error("Wrong starting site for two-site gate application.")
    orthogonalize!(psi, j1)
    j2 = j1 + 1
    A = (psi[j1] * psi[j2]) * G2
    noprime!(A)
    linds = uniqueinds(psi[j1], psi[j2])
    psi[j1], S, psi[j2], spec = svd(A, linds; cutoff=cutoff, maxdim=maxdim)
    psi[j2] *= S
    set_ortho_lims!(psi, j2:j2)
    return spec.truncerr
end

function apply3!(G3::ITensor, psi::MPS, j2::Int; cutoff::Real=1e-14, maxdim::Int=2*maxlinkdim(psi))
    """
    Apply three adjacent site gate `G3` to the MPS `psi` at sites `j2-1`, `j2`, and `j2+1` inplace.
    """
    (j2 <= 1 || j2 >= length(psi)) && error("Wrong middle site for three-site gate application.")
    orthogonalize!(psi, j2)
    s = siteind(psi, j2)
    j1, j3 = j2 - 1, j2 + 1
    A = (psi[j1] * psi[j2] * psi[j3]) * G3
    noprime!(A)
    linds = uniqueinds(psi[j1], psi[j2])
    psi[j1], S12, B, spec12 = svd(A, linds; cutoff=cutoff, maxdim=maxdim)
    B *= S12
    linds23 = (commonind(psi[j1], B), s)
    psi[j2], S23, psi[j3], spec23 = svd(B, linds23; cutoff=cutoff, maxdim=maxdim)
    psi[j2] *= S23
    set_ortho_lims!(psi, j2:j2)
    return spec12.truncerr + spec23.truncerr
end
