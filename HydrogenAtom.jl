"""
    HydrogenWaveFunc(n, l, m)

Get electron wave function in hydrogen atom.

Note that this wave function is using spherical coordinate system (r,θ,φ).
And Unit of length is Bohr radius α₀, which is defined as (4 π ϵ₀ ħ^2)/(mₑ e^2).

# Arguments
- `n::Integer`: principal quantum number, it is lager than 0.
- `l::Integer`: Azimuthal quantum number, it is lager than or equal to 0 and smaller than `n`.
- `m::Integer`: Magnetic quantum number, it's absolute value is smaller than or equal to `l`.

# Examples
```julia-repl
julia> ψ = HydrogenWaveFunc(4, 1, 1);

julia> ψ(5, π/2, 3π/2)
4.584892102393959e-19 + 0.002495899011092363im
```
"""
function HydrogenWaveFunc(n::Integer, l::Integer, m::Integer)
    if ~(n > 0  &&  0 <= l < n  &&  abs(m) <= l)
        error("invalid quantum number: n=$n, l=$l, m=$m")
    end

    # ┌ get wave function
    # │┌ Spherical Harmonics
    # ││┌ Associated Legendre Polynomial
    # │││┌ coeffients of Legendre Polynomial
    LegPolycoef = zeros(l + 1)
    for k ∈ 0 : l÷2
        LegPolycoef[1 + 2k] = (-1)^k * factorial(2(l - k)) /
            (factorial(k) * factorial(l - k) * factorial(l - 2k))
    end
    LegPolycoef ./= 2^l
    # │││└
    # │││┌ coeffients of m-th derivative of Legendre Polynomial
    for _ ∈ 1 : abs(m)
        LegPolycoef = LegPolycoef[1 : length(LegPolycoef)-1] .*
            (length(LegPolycoef)-1 : -1 : 1)
    end
    # │││└
    tmp1 = abs(m) / 2
    AssLeg = x -> begin
        result = 0.
        for c ∈ LegPolycoef        # calculate Polynomial
            result = result * x + c
        end
        return (1 - x^2)^tmp1 * result
    end
    # ││└
    # ││┌ normalize factor of Spherical Harmonics
    norm_lm = √((l + 0.5) / 2π  *  (factorial(l - abs(m)) / factorial(l + abs(m))))
    if m > 0
        norm_lm *= (-1)^m
    end
    # ││└
    Y = (θ, φ) -> norm_lm * AssLeg(cos(θ)) * exp(im * m * φ)
    # │└
    # │┌ Radial Function
    nᵣ = n - l - 1          # radial quantum number
    # ││┌ Confluent Hypergeometric Function ₁F₁
    # │││┌ coeffients of ₁F₁
    a = l + 1 - n; c = 2(l + 1)     # ₁F₁(a, c, z)
    F11Polycoef = zeros(nᵣ + 1)
    F11Polycoef[end] = 1
    for k ∈ 0 : nᵣ - 1
        F11Polycoef[nᵣ - k] = (a + k) / ((k + 1) * (k + c)) * F11Polycoef[nᵣ - k + 1]
    end
    # │││└
    F11 = z -> begin
        result = 0.
        for c ∈ F11Polycoef        # calculate Polynomial
            result = result * z + c
        end
        return result
    end
    # ││└
    # ││┌ normalize factor of Radial Function
    α = 2 / n
    norm_nl = 2 / factorial(2l + 1) * √(factorial(n + l) / factorial(nᵣ))
    norm_nl *= α^l
    norm_nl /= n^2          # I don't know how this appeared.
    # ││└
    R = r -> norm_nl * exp(-r / n) * r^l * F11(α * r)
    # │└
    return (r, θ, φ) -> R(r) * Y(θ, φ)
    # └
end


"""
    Cartesian2Spherical(x, y, z)

Tranform Cartesian coordinate system (x,y,z) to Spherical coordinate system (r,θ,φ).
"""
function Cartesian2Spherical(x::Real, y::Real, z::Real)
    r = √(x^2 + y^2 + z^2)
    (r ≈ 0) && (return 0., 0., 0.)
    φ = (x ≈ 0) ? ((y < 0) ? π/2 : -π/2) : atan(y / x)
    (x < 0) && (φ += π)
    (φ < 0) && (φ += 2π)
    return r, acos(z / r), φ
end

"""
    Spherical2Cartesian(r, θ, φ)

Tranform Spherical coordinate system (r,θ,φ) to Cartesian coordinate system (x,y,z).
"""
function Spherical2Cartesian(r::Real, θ::Real, φ::Real)
    sθ = sin(θ)
    return r * sθ * cos(φ), r * sθ * sin(θ), r * cos(θ)
end


###############################################################################


using Plots, ColorSchemes, PlotThemes, Printf
theme(:lime)
Plots.scalefontsizes(2.)


const figwidth = 1920
const figheight = 1080

const secperwave = 5

nlmz = (((1,0,0),4), ((2,0,0),10), ((2,1,0),14), ((2,1,1),14), ((3,0,0),20),
        ((3,1,0),26), ((3,1,1),26), ((3,2,0),26), ((3,2,1),26), ((3,2,2),26),
        ((4,0,0),34), ((4,1,0),40), ((4,1,1),40), ((4,2,0),44), ((4,2,1),44),
        ((4,2,2),44), ((4,3,0),44), ((4,3,1),44), ((4,3,2),44), ((4,3,3),44))


function l_name(l::Integer)
    (l == 0) && (return 's')
    (l == 1) && (return 'p')
    (l == 2) && (return 'd')
    (l < 7) && (return 'c' + l)
    return 'd' + l
end

function showwave(n::Integer, l::Integer, m::Integer, zp::Real)
    println("recall showwave")
    ψ = HydrogenWaveFunc(n, l, m)

    xp = zp * (figwidth / figheight)
    xrange = collect(range(-xp, xp, length = figwidth))
    zrange = collect(range(-zp, zp, length = figheight))

    w = Matrix{Float64}(undef, figheight, figwidth)
    for zidx ∈ 1 : figheight, xidx ∈ 1 : figwidth
        x = xrange[xidx]; z = zrange[zidx]
        w[zidx, xidx] = abs2(ψ(Cartesian2Spherical(x, 0, z)...))
    end

    fig = heatmap(xrange, zrange, w, c = cgrad(:gnuplot, scale = :exp),
        cbar = false, title = (), size = (figwidth, figheight))
    title!(fig, "$(n)$(l_name(l)), m=" * ((m==0) ? '0' : "±$(m)"))
    return fig
end

const fps = 24


function main()
    idx = 1
    n, l, m = nlmz[idx][1]
    fig = showwave(nlmz[idx][1]..., nlmz[idx][2])
    anim = @animate for frameidx ∈ 1 : UInt64(floor((fps * secperwave * length(nlmz))))
        if frameidx > (fps * secperwave * idx)
            idx += 1
            fig = showwave(nlmz[idx][1]..., nlmz[idx][2])
        end
        fig
    end

    println("animate frames storage in ", anim.dir)
    mp4(anim, "anim.mp4", fps = fps)
end

main()
