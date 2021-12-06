# More general function for electron orbit
# see more: https://github.com/nyasyamorina/trash-bin/blob/main/HydrogenAtom.jl


const ħ = 1.0545718e-34                         # (J ⋅ s)
const mₑ = 9.10938356e-31                       # (kg)
const e = 1.602176634e-19                       # (C)
const ε₀ = 8.854187817e-12                      # (F ⋅ m⁻¹)

const E₁ = -mₑ * e^4 / (32π^2 * ε₀^2 * ħ^2) / 1.60217662e-19    # (eV)
const α₀ = 4π * ε₀ * ħ^2 / (mₑ * e^2) * 1e10                    # (Å)


###############################################################################

# New features will probably be added in the future.

mutable struct Cartesian3D
    x::Float64
    y::Float64
    z::Float64
end

mutable struct Spherical3D
    r::Float64
    θ::Float64
    φ::Float64
end


@fastmath function Spherical3D(c::Cartesian3D)
    r = √(c.x^2 + c.y^2 + c.z^2)
    (r ≈ 0) && (return Spherical3D(0., 0., 0.))
    φ = (c.x ≈ 0) ? ((x≥0 ⊻ y>0) ? π/2 : -π/2) : atan(c.y / c.x)
    (c.x < 0) && (φ -= π)
    (φ < 0) && (φ += 2π)
    return Spherical3D(r, acos(c.z / r), φ)
end

@fastmath function Cartesian3D(c::Spherical3D)
    sinθ = sin(c.θ)
    return Cartesian3D(c.r * sinθ * cos(c.φ), c.r * sinθ * cos(c.φ), c.r * cos(c.θ))
end


###############################################################################


struct ElectronOrbit
    n::Int64
    l::Int64
    m::Int64
    E::Float64
    halfm::Float64
    AssLegPolycoef::Vector{Float64}
    Nₗₘ::Float64
    F11Polycoef::Vector{Float64}
    Nₙₗ::Float64
end

@fastmath function ElectronOrbit(n::Integer, l::Integer, m::Integer)
    absm = abs(m)
    @assert n > 0 && 0 ≤ l < n && absm ≤ l

    LegPolycoef = zeros(l + 1)
    for k ∈ 0:l÷2
        LegPolycoef[1+2k] = (-1)^k * factorial(2(l-k)) /
            (factorial(k) * factorial(l-k) * factorial(l-2k))
    end
    LegPolycoef .*= 2.0^-l
    for _ ∈ 1:absm
        LegPolycoef = LegPolycoef[1:length(LegPolycoef)-1] .*
            (length(LegPolycoef)-1:-1:1)
    end
    norm_lm = √(l+0.5) / 2π * (factorial(l-absm) / factorial(l+absm))
    if m > 0
        norm_lm *= (-1)^m
    end

    a = l + 1 - n
    c = 2(l + 1)
    F11Polycoef = zeros(n - l)
    F11Polycoef[end] = 1
    for k ∈ 0:n-l-2
        F11Polycoef[n-l-k-1] = (k+a) / ((k+1) * (k+c)) * F11Polycoef[n-l-k]
    end
    norm_nl = 2 / (n^2 * factorial(2l+1) * α₀) * √(factorial(n+l) / (factorial(n-l-1) * α₀))

    return ElectronOrbit(
        n, l, m, E₁ / n^2,
        absm/2, LegPolycoef, norm_lm,
        F11Polycoef, norm_nl
    )
end


@fastmath function Poly(Polycoef::Vector{T}, x::T) where T<:Number
    result = zero(T)
    for c ∈ Polycoef
        result = result * x + c
    end
    return result
end

@fastmath function AssLegPoly(H::ElectronOrbit, x::Real)       # x ∈ [-1,1]
    return (1 - x^2)^H.halfm * Poly(H.AssLegPolycoef, x)
end

@fastmath function SphericalHarmonic(H::ElectronOrbit, θ::Real, φ::Real)      # θ ∈ [0,π], φ ∈ [0,2π)
    return  H.Nₗₘ * AssLegPoly(H, cos(θ)) * cis(H.m * φ)
end

@fastmath function RadialFunc(H::ElectronOrbit, r::Real)       # r ∈ [0,+∞)
    ρ = 2r / (α₀*H.n)
    return H.Nₙₗ * exp(-0.5ρ) * ρ^H.l * Poly(H.F11Polycoef, ρ)
end

@fastmath function (H::ElectronOrbit)(p::Spherical3D)::ComplexF64
    return RadialFunc(H, p.r) * SphereicalHarmonic(H, p.θ, p.φ)
end

@fastmath function Pr(H::ElectronOrbit, p::Spherical3D)::Float64
    return (p.r * RadialFunc(H, p.r) * H.Nₗₘ * AssLegPoly(H, cos(p.θ)))^2 * sin(p.θ)
end

function (H::ElectronOrbit)(p::Cartesian3D)
    return H(Spherical3D(p))
end

function (H::ElectronOrbit)(x::Real, y::Real, z::Real)
    return H(Cartesian3D(x, y, z))
end

function Pr(H::ElectronOrbit, p::Cartesian3D)
    return Pr(H, Spherical3D(p))
end

function Pr(H::ElectronOrbit, x::Real, y::Real, z::Real)
    return Pr(H, Cartesian3D(x, y, z))
end
