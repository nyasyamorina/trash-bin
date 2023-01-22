""" Special Relativity Visulization"""

# this is a module for color space conversion, you can do the same thing with `Colors` module.
(@isdefined nyascolor) || include("nyascolor.jl")

using Main.nyascolor, FileIO, VideoIO, CUDA, ProgressMeter, Printf
import ColorTypes, FixedPointNumbers, JLD2, FreeTypeAbstraction


############################################################################### gpu stuff


const USE_CPU = ~CUDA.functional(true) || false
CUDA.allowscalar(false)


###############################################################################  geometry stuff


"Calculate the dot product of two vector"
dot(uᵅ, vᵅ) = sum(uᵅ .* conj.(vᵅ))
const ⋅ = dot

"Calculate the cross product of two 3D-vector"
cross((u¹, u², u³), (v¹, v², v³)) = begin
    u² * v³ - v² * u³,
    u³ * v¹ - v³ * u¹,
    u¹ * v² - v¹ * u²
end
const × = cross

"Calculate the length of the vector in Cartesian coordinates"
veclen(vᵅ) = sqrt(sum(abs2.(vᵅ)))

"Normalize a vector in Cartesian coordinates"
normalize(vᵅ) = vᵅ ./ veclen(vᵅ)

"Return both the length of the vector and the normalized vector"
lennorm(vᵅ) = begin
    len = veclen(vᵅ)
    return len, vᵅ ./ len
end


"""
    decompose(xᵅ, pᵅ)

Decompose a high dimension vector to a 2-D vector.

The `xᵅ` is a x-axis vector of 2-D Cartesian coordinates,
and return the normalized y-axis vector and the `qᵅ` which `pᵅ` at 2-D plane.

It may cause numerical instability when `xᵅ` and `pᵅ` are close to parallel.
And the `xᵅ` should be normalized.
"""
decompose(xᵅ, pᵅ) = begin
    q¹ = xᵅ ⋅ pᵅ
    q², yᵅ = lennorm(pᵅ .- q¹ .* xᵅ)
    yᵅ, (q¹, q²)
end

"The inverse function of `decompose`[@ref]"
compose(xᵅ, yᵅ, (q¹, q²)) = q¹ .* xᵅ .+ q² .* yᵅ


"Convert a poinnt in 3-D Cartesian coordinates into spherical coordinates"
tospherical((x, y, z)) = begin
    r = sqrt(x^2 + y^2 + z^2)
    r, acos(z / r), atan(y, x)
end


"""
    CameraLight(im_size, view_d, fov[; up])

Calculate the incoming light direction from the index on image.

The view direction `d` and view up direction `up` should be normalized.
"""
struct CameraLight
    size::NTuple{2, Int64}
    udv::NTuple{3, NTuple{3, Float64}}

    CameraLight((h, w), d, fov; up = (0., 0., 1.)) = begin
        u = d × up
        v = u × d
        a = tan(0.5fov)veclen(d)
        new((h, w), (a .* u, d, (h/w * a) .* v))
    end
end
(cl::CameraLight)(idx::CartesianIndex{2}) = cl(idx[2], idx[1])
(cl::CameraLight)(u, v) = begin
    uᵢ = 2(u - 0.5) / cl.size[2] - 1
    vᵢ = 1 - 2(v - 0.5) / cl.size[1]
    uᵢ .* cl.udv[1] .+ cl.udv[2] .+ vᵢ .* cl.udv[3]
end


###############################################################################  special relativity stuff

"light speed"
const c = 299_792_45

"""
    fov_shift(qᵅ, β)

Shift the light direction @ "movement" observer to the direction @ "stationary observer".

# Arguments
`qᵅ`: 2-d space coord, `q¹` should be the direction of motion
`β`: the speed of "movement" observer
"""
fovshift((q¹, q²), β) = (q¹ - β, q² * sqrt(1 + β^2))


"Calculate the spectral shift of the incoming light relative to \"movement\" observer"
spectralshift(β, cosθ) = sqrt(1 - β^2) / (1 + β * cosθ) - 1


"Lorentz transformation"
L((x⁰, x¹), β) = (x⁰ - β * x¹, x¹ - β * x⁰) ./ sqrt(1 - β^2)


###############################################################################  color stuff


const RGBN0f8 = ColorTypes.RGB{FixedPointNumbers.N0f8}


"Reducing mode of RGB color, it should be `:linear_max` or `:clamp`, otherwise doesn't reduce"
const REDUCE_COLOR = :clamp

"Reduce the RGB color to in range [0, 1], see also `REDUCE_COLOR`[@ref]"
reduce(rgb) = if REDUCE_COLOR == :linear_max
    max.(rgb ./ max(rgb..., 1), 0)
elseif REDUCE_COLOR == :clamp
    min.(max.(rgb, 0), 1) # ?gpu_err clamp.(rgb, 0, 1)
else
    rgb
end


"Convert temperatrue `T`(K) to mired `M`(MK⁻¹)"
mired(T) = 1_000_000 / T

const _c₂ = 14.387768775039339

"Planck's black body radiation, the wavelength `λ`(μm) & mired `M`(MK⁻¹)"
B(λ, M) = 1 / ((0.001λ)^5 * (exp(_c₂ * M / λ) - 1))     # `0.001` makes the result not so close to 0.

"Get the color (@ CIEXYZ) of black body @ mired `M` with luminance `L` (@ `z=0`) and spectral shift `z`"
function blackbody_XYZ((L, M); z = 0.)
    X = Y = Z = 0.
    k = 1 / (1 + z)
    for λ ∈ #=360:830=# 360:752     # less sum makes faster, & rel err < 1/255
        idx = λ - 359
        I = B(k * λ, M)
        X += CIEXYZ.x_data[idx] * I
        Y += CIEXYZ.y_data[idx] * I
        Z += CIEXYZ.z_data[idx] * I
    end
    (X, Y, Z) .* (k * L / Y)
end

"Pipline to get color (@ sRGB) of black body, see also `blackbody_XYZ`[@ref]"
blackbody_sRGB((L, M); z = 0.) = blackbody_XYZ((L, M); z) |> CIEXYZ.tolinear_sRGB |> reduce |> linear_sRGB.tosRGB


"Render `string` using `fontface` with `pixelsize`, and return a `Matrix{UInt8}`"
function renderline(str, face, pixsize)   # rewrite from `FreeTypeAbstraction.renderstring!`
    chars = str isa AbstractVector ? str : collect(str)
    len = length(chars)
    len ≠ 0 || return Matrix{UInt8}(undef, 0, 0)
    bitmaps = Vector{Matrix{UInt8}}(undef, len)
    somenums = Vector{NTuple{5, Int}}(undef, len)
    ymax = ymin = xtotal = 0
    for (idx, char) ∈ enumerate(chars)
        bitmap, metricf = FreeTypeAbstraction.renderface(face, char, pixsize)
        bitmaps[idx] = bitmap
        bx, by = Int.(metricf.horizontal_bearing)
        ax, _  = Int.(metricf.advance)
        sx, sy = Int.(metricf.scale)
        ymax = max(ymax, by)
        ymin = min(ymin, by - sy)
        xtotal += idx ≠ len ? ax : bx + sx
        somenums[idx] = (bx, by, ax, sx, sy)
    end
    px = -somenums[1][1]
    py = ymax
    im = fill(zero(UInt8), ymax - ymin, xtotal + px)
    for (idx, (bitmap, (bx, by, ax, sx, sy))) ∈ enumerate(zip(bitmaps, somenums))
        oy = py - by
        ox = px + bx
        for y ∈ 1:sy, x ∈ 1:sx          # should be in range
            ov = UInt16(im[oy + y, ox + x])
            bv = UInt16(bitmap[x, y])
            v = ov + bv - ov * bv ÷ 0xFF
            im[oy + y, ox + x] = UInt8(v)
        end
        px += ax
    end
    return im
end


###############################################################################  utils stuff


# some useful 16:9 ratio resolution
const _480p = (480, 854)
const _720p = (720, 1280)
const _1080p = (1080, 1920)
const _1440p = (1440, 2560)
const _2160p = (2160, 3840)


"Calculate the position on the panorama image for given spherical angel"
panorama_uv((h, w), θ, φ) = mod2pi(φ) * w / 2π, θ * h / π

"Coefficints of cubic interpolation"
cubiccoef(x) = begin
    half_x = 0.5x
    x_over_3 = x / 3
    c₋₂ = ((1 - x_over_3) * half_x - 1/3) * x
    c₋₁ = ((half_x - 1) * x - 0.5) * x + 1
    c₀ = ((1 - x) * half_x + 1) * x
    c₁ = (half_x*x - 0.5) * x_over_3
    c₋₂, c₋₁, c₀, c₁
end


###############################################################################  main


"Path of the jld2 file of panorama image, it storges a Matrix of (luminance, mired) tuple"
const L_CCT_path = joinpath(@__DIR__, "eso0932a_L_CCT.jld2")

"Path of font or name of font face"
const font = "Consolas"


"Get the (luminance, mired) @ image idx `(u, v)` in jld2 of panorama `a`"
function get_L_CCT(a, (u, v))
    h, w = size(a)
    uᵢ_f, uᵣ = divrem(u + 0.5, 1)
    vᵢ_f, vᵣ = divrem(v + 0.5, 1)
    uᵢ = mod(floor(Int, uᵢ_f), w) + 1
    vᵢ = mod(floor(Int, vᵢ_f), h) + 1
    tmp = (1, -1, -2)
    uᵢ₊₁, uᵢ₋₁, uᵢ₋₂ = mod.((uᵢ - 1) .+ tmp, w) .+ 1
    vᵢ₊₁, vᵢ₋₁, vᵢ₋₂ = abs.(mod.((vᵢ - h) .+ tmp, 2(h - 1)) .- (h - 1)) .+ 1
    u₋₂, u₋₁, u₀, u₁ = cubiccoef(uᵣ)
    v₋₂, v₋₁, v₀, v₁ = cubiccoef(vᵣ)
    L = v₋₂ * (u₋₂ * a[vᵢ₋₂, uᵢ₋₂][1] + u₋₁ * a[vᵢ₋₂, uᵢ₋₁][1] + u₀ * a[vᵢ₋₂, uᵢ][1] + u₁ * a[vᵢ₋₂, uᵢ₊₁][1]) +
        v₋₁ * (u₋₂ * a[vᵢ₋₁, uᵢ₋₂][1] + u₋₁ * a[vᵢ₋₁, uᵢ₋₁][1] + u₀ * a[vᵢ₋₁, uᵢ][1] + u₁ * a[vᵢ₋₁, uᵢ₊₁][1]) +
        v₀  * (u₋₂ * a[vᵢ  , uᵢ₋₂][1] + u₋₁ * a[vᵢ  , uᵢ₋₁][1] + u₀ * a[vᵢ  , uᵢ][1] + u₁ * a[vᵢ  , uᵢ₊₁][1]) +
        v₁  * (u₋₂ * a[vᵢ₊₁, uᵢ₋₂][1] + u₋₁ * a[vᵢ₊₁, uᵢ₋₁][1] + u₀ * a[vᵢ₊₁, uᵢ][1] + u₁ * a[vᵢ₊₁, uᵢ₊₁][1])
    return (L, a[vᵢ, uᵢ][2])
end


"Render a frame"
function renderframe(a, im_size, cam_d, fov, β; cam_up = (0., 0., 1.))
    cam_d = normalize(cam_d)
    camlight = CameraLight(im_size, cam_d, fov; up = normalize(cam_up))
    a_size = size(a)
    a_xpu = USE_CPU ? a : CuArray(a)

    _core = (idxx, idxy) -> begin
        light = camlight(idxx, idxy)
        yᵅ, qᵅ = decompose(cam_d, light)
        qᵅ = fovshift(normalize(qᵅ), β)
        z = spectralshift(β, qᵅ[1] / veclen(qᵅ))
        light = compose(cam_d, yᵅ, qᵅ)
        _, θ, φ = tospherical(light)
        uvₐ = panorama_uv(a_size, θ, φ)
        L_CCT = get_L_CCT(a_xpu, uvₐ)
        blackbody_sRGB(L_CCT; z)
    end

    im = fill(RGBN0f8(0, 0, 0), im_size)
    if USE_CPU
        Threads.@threads for idx ∈ CartesianIndices(im)
            idxy, idxx = Tuple(idx)
            im[idx] = _core(idxx, idxy) |> sRGB.toRGBN0f8
        end
    else
        idxx = CuArray(collect(1:im_size[2])')
        idxy = CuArray(collect(1:im_size[1]))
        im_tuple = _core.(idxx, idxy) |> Array
        im .= sRGB.toRGBN0f8.(im_tuple)
    end
    return im
end
#renderframe(L_CCT_jld2, _1080p, (0., 1., 0.), deg2rad(100), 0.8)

"Put infomation of speed @ left bottom of image"
function puttext!(im, face, fontsize, β; α = 0.75)
    β_text = "beta = " * (@sprintf "%+.6f" β)   # using "beta" beacuse some font doesn't support "β"
    v_text = @sprintf "%+.2f" 0.001c * β
    v_text = "   v = " * ' '^(9 - length(v_text)) * v_text * "km/s"
    β_im = renderline(β_text, face, fontsize)
    v_im = renderline(v_text, face, fontsize)
    β_h, β_w = size(β_im)
    v_h, v_w = size(v_im)
    gap = 12fontsize ÷ 22
    text_im = zeros(UInt8, 3gap + β_h + v_h, 2gap + max(β_w, v_w))
    text_im[gap + 1:gap + β_h, gap + 1:gap + β_w] .= β_im
    text_im[2gap + β_h + 1:2gap + β_h + v_h, gap + 1:gap + v_w] .= v_im
    text_h, text_w = size(text_im)
    section = @view im[size(im, 1) - text_h + 1:end, 1:text_w]
    Threads.@threads for idx ∈ CartesianIndices(text_im)
        rgb = section[idx] |> sRGB.fromColorant |> sRGB.tolinear_sRGB
        v = text_im[idx] / 0xFF
        rgb = ((1 - v) * (1 - α)) .* rgb .+ v
        section[idx] = rgb |> linear_sRGB.tosRGB |> sRGB.toRGBN0f8
    end
end


function _renderframe!(a, yᵅ, qᵅ, cam_d, β, tmp1, tmp2, im)
    a_size = size(a)
    _core = (yᵝ, qᵝ) -> begin
        qᵝ = fovshift(qᵝ, β)
        z = spectralshift(β, qᵝ[1] / veclen(qᵝ))
        light = compose(cam_d, yᵝ, qᵝ)
        _, θ, φ = tospherical(light)
        uvₐ = panorama_uv(a_size, θ, φ)
        L_CCT = get_L_CCT(a, uvₐ)
        blackbody_sRGB(L_CCT; z)
    end
    if USE_CPU
        Threads.@threads for idx ∈ CartesianIndices(im)
            im[idx] = _core(yᵅ[idx], qᵅ[idx]) |> sRGB.toRGBN0f8
        end
    else
        tmp1 .= _core.(yᵅ, qᵅ)
        copy!(tmp2, tmp1)
        im .= sRGB.toRGBN0f8.(tmp2)
    end
end

function rendervideo(path, fps, runtime, a, im_size, cam_d, fov, acc, jump, fontface;
                     cam_up = (0., 0., 1.), encoder_options = (crf = 23, preset = "medium"))
    im = fill(RGBN0f8(0, 0, 0), im_size)
    fontsize = 40im_size[1] ÷ 1080
    dτ = 1 / (fps * jump)
    cam_d = normalize(cam_d)
    camlight = CameraLight(im_size, cam_d, fov; up = normalize(cam_up))
    a_size = size(a)
    a_xpu = USE_CPU ? a : CuArray(a)

    yᵅ = similar(a_xpu, NTuple{3, Float64}, im_size)
    qᵅ = similar(a_xpu, NTuple{2, Float64}, im_size)
    tmp1, tmp2 = if USE_CPU
        Threads.@threads for idx ∈ CartesianIndices(im)
            yᵝ, qᵝ = decompose(cam_d, camlight(idx))
            yᵅ[idx] = yᵝ
            qᵅ[idx] = normalize(qᵝ)
        end
        nothing, nothing
    else
        tmp3 = ((u, v) -> decompose(cam_d, camlight(u, v))).(
            CuArray(collect(1:im_size[2])'),
            CuArray(collect(1:im_size[1]))
        )
        yᵅ .= (x -> x[1]).(tmp3)
        qᵅ .= (x -> normalize(x[2])).(tmp3)
        similar(yᵅ), similar(im, eltype(yᵅ))
    end

    open_video_out(path, im; framerate = fps, encoder_options) do writer
        τ = 0.
        absxᵅ = (dτ, 0.)
        @showprogress for _ ∈ 0:(1/fps):runtime
            β = -absxᵅ[2] / absxᵅ[1]
            _renderframe!(a_xpu, yᵅ, qᵅ, cam_d, β, tmp1, tmp2, im)
            puttext!(im, fontface, fontsize, β)
            write(writer, im)
            for _ ∈ 1:jump
                absxᵅ = L(absxᵅ, dτ * acc(τ))
                τ += dτ
            end
        end
    end
end


(@isdefined L_CCT_jld2) || (const L_CCT_jld2 = JLD2.load_object(L_CCT_path))
fontface = isfile(font) ? FreeTypeAbstraction.FTFont(font) : FreeTypeAbstraction.findfont(font)
acc(τ) = (2.5π ≤ τ < 7.5π ? -1 : 1) * sin(0.4τ)^2

#rendervideo(joinpath(@__DIR__, "out.mp4"), 15, 10π, L_CCT_jld2, _480p, (0., 1., 0.), deg2rad(100), acc, 4, fontface)
rendervideo(joinpath(@__DIR__, "out.mp4"), 60, 10π, L_CCT_jld2, _2160p, (0., 1., 0.), deg2rad(100), acc, 4, fontface)
