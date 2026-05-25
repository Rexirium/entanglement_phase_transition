function applyn!(G::ITensor, psi::MPS; cutoff::Real=1e-14, maxdim::Int=4*maxlinkdim(psi), rev::Bool=false)
    js = findsites(psi, G)
    return applyn!(G, psi, js...; cutoff=cutoff, maxdim=maxdim, rev=rev)
end

function applyn!(G1::ITensor, psi::MPS, j::Int; cutoff::Real=1e-14, maxdim::Int=4*maxlinkdim(psi), rev::Bool=false)
    """
    Apply the gate `G1` to the MPS `psi` at site `j` inplace.
    """
    orthogonalize!(psi, j)
    psi[j] *= G1
    noprime!(psi[j])
    return 0.0
end

function applyn!(G2::ITensor, psi::MPS, j1::Int, j2::Int; cutoff::Real=1e-14, maxdim::Int=4*maxlinkdim(psi), rev::Bool=false)
    """
    Apply two adjacent site gate `G2` to the MPS `psi` at sites `j1` and `j1+1` inplace.
    """
    (j1<=0 || j1>= length(psi)) && error("Wrong starting site for two-site gate application.")
    ja, jb = rev ? (j2, j1) : (j1, j2)
    orthogonalize!(psi, ja)

    A = (psi[ja] * psi[jb]) * G2
    noprime!(A)
    indsab = uniqueinds(psi[ja], psi[jb])
    psi[ja], S, psi[jb], spec = svd(A, indsab; cutoff=cutoff, maxdim=maxdim)
    psi[jb] *= S

    replacetags!(psi[ja], "Link,u" => "Link,l=$j1")
    replacetags!(psi[jb], "Link,u" => "Link,l=$j1")
    set_ortho_lims!(psi, jb:jb)

    return spec.truncerr
end

function applyn!(G3::ITensor, psi::MPS, j1::Int, j2::Int, j3::Int; cutoff::Real=1e-14, maxdim::Int=4*maxlinkdim(psi), rev::Bool=false)
    """
    Apply three adjacent site gate `G3` to the MPS `psi` at sites `j2-1`, `j2`, and `j2+1` inplace.
    """
    (j2 <= 1 || j2 >= length(psi)) && error("Wrong middle site for three-site gate application.")
    ja, jb, jc = rev ? (j3, j2, j1) : (j1, j2, j3)
    jab, jbc = rev ? (j2, j1) : (j1, j2)
    orthogonalize!(psi, ja)
    s = siteind(psi, jb)

    A = (psi[ja] * psi[jb] * psi[jc]) * G3
    noprime!(A)

    indsab = uniqueinds(psi[ja], psi[jb])
    psi[ja], Sab, B, specab = svd(A, indsab; cutoff=cutoff, maxdim=maxdim)
    B *= Sab
    
    replacetags!(psi[ja], "Link,u" => "Link,l=$jab")
    replacetags!(B, "Link,u" => "Link,l=$jab")

    indsbc = (commonind(psi[ja], B), s)
    psi[jb], Sbc, psi[jc], specbc = svd(B, indsbc; cutoff=cutoff, maxdim=maxdim)
    psi[jc] *= Sbc
    
    replacetags!(psi[jb], "Link,u" => "Link,l=$jbc")
    replacetags!(psi[jc], "Link,u" => "Link,l=$jbc")
    set_ortho_lims!(psi, jc:jc)

    return specab.truncerr + specbc.truncerr
end

function applyn!(Gs::Vector{ITensor}, psi::MPS; cutoff::Real=1e-14, maxdim::Int=4*maxlinkdim(psi), rev::Bool=false)
    """
    Apply a vector of gates `Gs` to the MPS `psi` inplace.
    """
    truncerr = 0.0
    for G in Gs
        truncerr += applyn!(G, psi; cutoff=cutoff, maxdim=maxdim, rev=rev)
    end
    return truncerr
end

struct InfMPS <: AbstractMPS
    len_uc::Int
    Gammas::Vector{ITensor}
    Lambdas::Vector{ITensor}
end

function InfMPS(ss::Vector{<:Index}, state::Function)
    len_uc = length(ss)

    ls = [Index(1, "LLink,l=$n") for n in 1:len_uc]
    rs = [Index(1, "RLink,r=$n") for n in 1:len_uc]

    Γs = ITensor[]
    Λs = ITensor[]
    for n in 1 : len_uc
        sn = state(n) == "Up" ? 1 : 2
        nn = mod1(n+1, len_uc)

        Γ = ITensor(ls[n], ss[n], rs[n])
        Λ = ITensor(rs[n], ls[nn])

        Γ[ls[n]=>1, ss[n]=>sn, rs[n]=>1] = 1.0
        Λ[rs[n]=>1, ls[nn]=>1] = 1.0

        push!(Γs, Γ)
        push!(Λs, Λ)
        
    end
    return InfMPS(len_uc, Γs, Λs)
end