import ColorTypes
using StaticArrays, FileIO


# define `Vec3` & `RGB` types
const RGB = SVector{3, Float32}
const Vec3 = SVector{3, Float64}
RGB(gray::Number) = RGB(gray, gray, gray)

# define `Image` type & `save` function
const Image = Matrix{RGB}
FileIO.save(@nospecialize(path), im::Image) =
    FileIO.save(path, Array(reinterpret(ColorTypes.RGB{Float32}, im)))

# some useful functions
pow2(x::Number) = x * x
dot(u::Vec3, v::Vec3) = sum(u .* v)
dot(u, v) = begin; @nospecialize u v; Vec3(u) ⋅ Vec3(v); end
abs2(v::Vec3) = v ⋅ v
distance2(p, q) = abs2(q .- p)
normalize(v::Vec3) = v .* (1 / sqrt(abs2(v)))
normalize(@nospecialize v) = normalize(Vec3(v))
const ⋅ = dot


# color wheel function
_colorf(n, h) = begin
    k = mod(n + h, 6)
    k > 2 && (k = 4 - k)    # if (k > 2) then (k = 4 - k)
    clamp(k, 0, 1)
end
"Return the color in color wheel with angle `ϕ`"
colorwheel(ϕ) = begin
    h = 3.f0 / π * convert(Float32, ϕ)
    RGB(_colorf(n, h) for n ∈ (2, 0, 4))
end



"""
    Source(p[, a])

Source in static field.

# Arguments
- `p::Vec3`: the position of source.
- `a::ComplexF64=0`: the amplitude `A` and first phase `ϕ`, `a = Ae^(iϕ)`

See also [`A`](@ref), [`ϕ`](@ref), [`getsignal`](@ref), [`PhasedArray`](@ref).
"""
mutable struct Source
    p::Vec3
    a::ComplexF64
    Source(@nospecialize(p), a) = new(Vec3(p), a)
end
Source(@nospecialize p) = Source(Vec3(p), zero(ComplexF64))

"Return the amplitude of the source"
A(src::Source) = abs(src.a)
"Return the first phase of the source"
ϕ(src::Source) = angle(src.a)

"Get the signal (field) of the source `src` at position `p` and frequency `f`."
getsignal(src::Source, p::Vec3, f::Float64) = begin
    d² = distance2(src.p, p)
    src.a * cispi(-2f * sqrt(d²)) / d²   # cispi(x) = e^(πix)
end
getsignal(src::Source, @nospecialize(p), f::Float64) = getsignal(src, Vec3(p), f)


"""
    PhasedArray(f[, sources])

the phased array contains multi-source.

# Arguments
- `f::Float64`: the frequency of phased array.
- `sources::Vector{Source}=[]`: the sources in phased array.

See also [`Source`](@ref), [`getsignal`](@ref), [`backward!`](@ref), [`render_field`](@ref).
"""
struct PhasedArray
    f::Float64
    sources::Vector{Source}
end
PhasedArray(@nospecialize f) = PhasedArray(f, Source[])

"Get the signal (field) of phased array `pa` at position `p`."
getsignal(pa::PhasedArray, p::Vec3) = sum(src -> getsignal(src, p, pa.f), pa.sources)
getsignal(pa::PhasedArray, @nospecialize p) = getsignal(pa, Vec3(p))


"Reverse the light path from `srcs`->`pa` to `pa`->`srcs`. `c` is to avoid overflow or underflow."
backward!(pa::PhasedArray, srcs::Source...; c = 1.) = for receiver ∈ pa.sources
    receiver.a = sum(src -> c / getsignal(src, receiver.p, pa.f), srcs)
end



"""
    square_lattice(n)

Generate a square lattice in range [0, 1]².

Note: the length of return may different from `n`.

See also [`hexagonal_lattice`](@ref).
"""
function square_lattice(n::Int)
    s = floor(Int, sqrt(n))
    ps = Vector{Tuple{Float64, Float64}}(undef, s * s)
    idx = 0
    for k₁ ∈ 1:s, k₂ ∈ 1:s
        ps[idx += 1] = @. ((k₁, k₂) - 1) / (s - 1)
    end
    return ps
end

"""
    hexagonal_lattice(n)

Generate a hexagonal lattice in range [0, 1]².

Note: the length of return may different from `n`.

See also [`square_lattice`](@ref).
"""
function hexagonal_lattice(n::Int)
    s = floor(Int, sqrt(sqrt(0.75) * n + 0.0625) - 0.25)
    t = floor(Int, sqrt(1 / 3) * (2s + 1))
    ps = Vector{Tuple{Float64, Float64}}(undef, s * t)
    idx = 0
    for k₁ ∈ 1:t, k₂ ∈ 1:s
        y = sqrt(0.75) * (k₁ - 0.5t - 0.5) / (s - 0.5) + 0.5
        x = (k₂ - 1) / (s - 0.5)
        iseven(k₁) && (x += 0.5 / (s - 0.5))
        ps[idx += 1] = (x, y)
    end
    return ps
end



"""
    render_colorwheel(size[; hole, thick, fade])

Return color wheel image with `size`.

# Arguments
- `size::Tuple{Int, Int}`: the size of output image, it should be (height, width).
- `hole::Float64=0.4`: the radius of hole in the wheel.
- `thick::Float64=0.5`: the thickness of the wheel.
- `fade::Float64=0.02`: the distance of fade in and out.

"""
function render_colorwheel(@nospecialize size; hole::Float64 = 0.4,
                           thick::Float64 = 0.5, fade::Float64 = 0.02)
    im = Image(undef, size...)
    Threads.@threads for idx ∈ CartesianIndices(im)
        x = (2idx[2] - 1) / size[2] - 1
        y = 1 - (2idx[1] - 1) / size[1]
        l = sqrt(pow2(x) + pow2(y))
        alpha = 1 - clamp(abs(l - hole - 0.5thick) - 0.5thick + fade, 0, fade) / fade
        @inbounds im[idx] = colorwheel(atan(y, x)) .* alpha
    end
    return im
end


"""
    render_field(pa, size, tl, tr, bl[; render_phase, Amax, bright])

Render the filed of phased array `pa` and return.

# Arguments
- `pa::PhasedArray`: the phased array that generate the field.
- `size::Tuple{Int, Int}`: the size of output image, it should be (height, width).
- `tl::Vec3`: the top left corner position of image.
- `tr::Vec3`: the top right corner position of image.
- `bl::Vec3`: the bottom left corner position of image.
- `render_phase::Symbol=:both`: `:no` means only render amplitude,
`:only` means only render phase but not amplitude, `:both` means render both amplitude and phase.
- `Amax::Union{Float64, Missing}=missing`: the max amplitude that will clamp to,
if `missing` it will auto calculate this value, only work with `render_phase = :no/:both`.
- `bright::Float64=1`: is similar to `Amax`, but it can work with `Amax = nothing`.

"""
function render_field(pa::PhasedArray, size, tl, tr, bl; render_phase::Symbol = :both,
                      Amax::Union{Float64, Missing} = missing, bright::Float64 = 1.)
    @nospecialize size tl tr bl
    @assert render_phase ∈ (:no, :only, :both)
    o = Vec3(tl)
    u = Vec3(tr) .- o
    v = Vec3(bl) .- o

    signals = Matrix{ComplexF64}(undef, size...)
    Threads.@threads for idx ∈ CartesianIndices(signals)
        x = (idx[2] - 0.5) / size[2]
        y = (idx[1] - 0.5) / size[1]
        p = @. x * u + y * v + o    # `p` = position in 3D-space at pixel `idx`
        @inbounds signals[idx] = getsignal(pa, p)
    end

    brifunc = if render_phase === :only; A -> 1.f0; else
        amax = ismissing(Amax) ? (sum(abs, signals) / length(signals)) : abs(Amax)
        ka = convert(Float32, bright / amax)
        A -> clamp(ka * convert(Float32, A), 0, 1)#^0.7f0    # `^ 0.7` for beauty
    end

    render_kernel = if render_phase === :no
        signal -> RGB(brifunc(abs(signal)))
    elseif render_phase === :only
        signal -> colorwheel(angle(signal))
    else
        signal -> brifunc(abs(signal)) .* colorwheel(angle(signal))
    end

    return render_kernel.(signals)::Image
end



function example1()
    # test color wheel functional
    im = render_colorwheel((256, 256); hole = 0.7, thick = 0.2, fade = 0.03)
    save("out.png", im)
    @info "saved image at `$(abspath("out.png"))`"

    # args
    pa = PhasedArray(10)    # frequency
    r = 2.                  # position of sources ∈ [-r, r]
    n = 1000                # n of sources
    αs = [60, 135]          # directions (deg)

    # setup sources
    ϕs = (2pa.f * d) .* cos.(deg2rad.(αs))
    for k ∈ 1:n
        push!(pa.sources, Source((r - 2r / n * k, 0, 0), sum(cispi.(k .* ϕs))))
    end
    #backward!(pa, Source((1, 2, 0), 1), Source((-2, 2, 0), 1))     # backward test

    # render & save
    zoom = 30
    im = render_field(pa, (512, 512), (-zoom, 2zoom, 0), (zoom, 2zoom, 0), (-zoom, 0, 0);
                      render_phase = :no, bright = 1.)
    save("out.png", im)
    return nothing
end


function example2(closeup::Bool = false)
    # args
    pa = PhasedArray(10)                # frequency
    lattice = square_lattice(4900)      # choose sources lattice & n of sources
    r = 2                               # position of sources ∈ [-r, r]²
    dires = [(1, 1, 1), (-1, 0, 1)]     # directions (can be non-normalize)

    direns = normalize.(dires)
    for lp ∈ lattice
        p = Vec3((2r .* lp .- r)..., 0)
        push!(pa.sources, Source(p, sum(cispi(-2pa.f * (p ⋅ d)) for d ∈ direns)))
    end
    zoom = closeup ? 2.5 : 20000.
    z = closeup ? 0.00001 : 10000.
    im = render_field(pa, (512, 512), (-zoom, zoom, z), (zoom, zoom, z), (-zoom, -zoom, z);
                      render_phase = closeup ? :both : :no, bright = closeup ? 2. : 0.05)
    save("out.png", im)
    return nothing
end
