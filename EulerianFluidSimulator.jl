using CUDA, ColorTypes, FixedPointNumbers, Printf, FileIO, FFMPEG, ProgressMeter

CUDA.allowscalar(false)


""" the physical position at one cell in the grid

                 ↑ up speed
            +----o----+
            |         |
          ←-o    o    o-→
 left speed |  center | right speed
            +----o----+
                 ↓ down speed

The position of elements in `u` field is the center of left edge.
The position of elements in `v` field is the center of down edge.
The position of elements in other fields are the center of cell.
The right speed of the current cell is also the left speed of the right cell,
other speeds is in the same way."""


struct FluidGPU
    size::NTuple{2, Int}    # total size of grid
    Δc::Float32             # size of grid cell
    f::CuMatrix{Bool}       # is cell fluid?
    u::CuMatrix{Float32}    # horizontal speed
    v::CuMatrix{Float32}    # vertical speed
    p::CuMatrix{Float32}    # pressure
    s::CuMatrix{Float32}    # smoke
    _tmp::NTuple{5, CuMatrix{Float32}}  # some buffer for calculation

    function FluidGPU(size, Δc)
        """Extra grid should be padded outside the simulation area,
        so the size of `f` and `s` fields are `(H+2,W+2)` instead of `(H,W)`."""
        H, W = size
        f = CUDA.trues(H + 2, W + 2)
        u = CUDA.zeros(Float32, H, W + 1)
        v = CUDA.zeros(Float32, H + 1, W)
        p = CUDA.zeros(Float32, H + 2, W + 2)
        s = CUDA.zeros(Float32, H + 2, W + 2)
        return new(
            size, Δc, f, u, v, p, s,
            (similar(u), similar(u), similar(v), similar(v), similar(s))
        )
    end
end


function applyincompressibility(f::FluidGPU, Δt, n)
    "the fluid is incompressible means the fluid flows into the cell is equal to flow out of the cell."
    H, W = f.size
    p_ = Float32(f.Δc / Δt)     # pressure change

    core = (f, u, v, p, Δu₁, Δu₂, Δv₁, Δv₂) -> @inbounds begin
        linear = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        w, h = divrem(linear - 1, H) .+ (1, 1)      # get the index of cell at current thread
        w ≤ W || return                             # must be in bounds
        if f[h + 1, w + 1]                      # current cell shound be fluid
            fl = Int(f[h + 1, w])
            fd = Int(f[h, w + 1])
            fr = Int(f[h + 1, w + 2])
            fu = Int(f[h + 2, w + 1])
            # the divergence of each speed which connect the fluid cell nearby
            d = -(u[h, w] + v[h, w] - u[h, w + 1] - v[h + 1, w]) / (fl + fd + fr + fu)
            p[h + 1, w + 1] -= d * p_           # update pressure
            # record the speed change due to incompressibility
            Δu₁[h, w] = d * fl
            Δv₁[h, w] = d * fd
            Δu₂[h, w + 1] = -d * fr
            Δv₂[h + 1, w] = -d * fu
        else
            Δu₁[h, w] = 0
            Δv₁[h, w] = 0
            Δu₂[h, w + 1] = 0
            Δv₂[h + 1, w] = 0
        end
        return
    end
    f._tmp[1][:, W + 1] .= f._tmp[2][:, 1] .= 0
    f._tmp[3][H + 1, :] .= f._tmp[4][1, :] .= 0
    # compile the cuda kernel function
    args = (f.f, f.u, f.v, f.p, f._tmp[1:4]...)
    kernel = @cuda launch = false core(args...)
    threads = min(H * W, launch_configuration(kernel.fun).threads)
    blocks = cld(H * W, threads)
    for _ ∈ 1:n                                         # solve incompressibility multiple time
        CUDA.@sync kernel(args...; threads, blocks)     # start the cuda kernel function
        # update speed
        CUDA.@sync begin
            f.u .+= f._tmp[1] .+ f._tmp[2]
            f.v .+= f._tmp[3] .+ f._tmp[4]
        end
    end
end

function applyflow(f::FluidGPU, Δt)
    """To apply flow, first calculate the physical position `p` and the speed `v`
    of each elements in each fields, and calculate the previous position `pₜ = p - Δt * v`,
    then the next state at `p` in the field equal to the current state at `pₜ` in the field.
    And calculation at the physical position can be simplify to at the index."""
    H, W = f.size
    ΔtΔc = Float32(Δt / f.Δc)

    core_u = (f, u, v, uₜ) -> @inbounds begin
        linear = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        w, h = divrem(linear - 1, H) .+ (1, 1)          # get the index
        w ≤ W + 1 || return
        uₜ[h, w] = if f[h + 1, w + 1] && f[h + 1, w]
            # get the physical speed at the position of this element in the `u` field
            Δx = u[h, w]
            Δy = if w == 1
                0.5f0 * (v[h, 1] + v[h + 1, 1])
            elseif w == W + 1
                0.5f0 * (v[h, W] + v[h + 1, W])
            else
                0.25f0 * (v[h, w] + v[h + 1, w] + v[h, w - 1] + v[h + 1, w - 1])
            end
            # the "speed of index" given by `v / Δc`
            hₜ, Δh = divrem(h - ΔtΔc * Δy, 1)
            wₜ, Δw = divrem(w - ΔtΔc * Δx, 1)
            # sample the `u` field at "previous index" using bilinear interpolation
            h₁ = clamp(Int(hₜ), 1, H)
            w₁ = clamp(Int(wₜ), 1, W + 1)
            h₂ = clamp(Int(hₜ) + 1, 1, H)
            w₂ = clamp(Int(wₜ) + 1, 1, W + 1)
            (1 - Δh) * ((1 - Δw) * u[h₁, w₁] + Δw * u[h₁, w₂]) +
            Δh * ((1 - Δw) * u[h₂, w₁] + Δw * u[h₂, w₂])
        else
            u[h, w]
        end
        return
    end
    core_v = (f, u, v, vₜ) -> @inbounds begin
        # the same thing happened in `core_u`
        linear = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        w, h = divrem(linear - 1, H + 1) .+ (1, 1)
        w ≤ W || return
        vₜ[h, w] = if f[h + 1, w + 1] && f[h, w + 1]
            Δx = if h == 1
                0.5f0 * (u[1, w] + u[1, w + 1])
            elseif h == H + 1
                0.5f0 * (u[H, w] + u[H, w + 1])
            else
                0.25f0 * (u[h, w] + u[h, w + 1] + u[h - 1, w] + u[h - 1, w + 1])
            end
            Δy = v[h, w]
            hₜ, Δh = divrem(h - ΔtΔc * Δy, 1)
            wₜ, Δw = divrem(w - ΔtΔc * Δx, 1)
            h₁ = clamp(Int(hₜ), 1, H + 1)
            w₁ = clamp(Int(wₜ), 1, W)
            h₂ = clamp(Int(hₜ) + 1, 1, H + 1)
            w₂ = clamp(Int(wₜ) + 1, 1, W)
            (1 - Δh) * ((1 - Δw) * v[h₁, w₁] + Δw * v[h₁, w₂]) +
            Δh * ((1 - Δw) * v[h₂, w₁] + Δw * v[h₂, w₂])
        else
            v[h, w]
        end
        return
    end
    core_s = (f, u, v, s, sₜ) -> @inbounds begin
        # the same thing happened in `core_u`
        linear = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        w, h = divrem(linear - 1, H + 2) .+ (1, 1)
        w ≤ W + 2 || return
        sₜ[h, w] = if 2 ≤ h ≤ H + 1 && 2 ≤ w ≤ W + 1 && f[h, w]
            Δx = 0.5f0 * (u[h - 1, w - 1] + u[h - 1, w])
            Δy = 0.5f0 * (v[h - 1, w - 1] + v[h, w - 1])
            hₜ, Δh = divrem(h - ΔtΔc * Δy, 1)
            wₜ, Δw = divrem(w - ΔtΔc * Δx, 1)
            h₁ = clamp(Int(hₜ), 1, H + 2)
            w₁ = clamp(Int(wₜ), 1, W + 2)
            h₂ = clamp(Int(hₜ) + 1, 1, H + 2)
            w₂ = clamp(Int(wₜ) + 1, 1, W + 2)
            (1 - Δh) * ((1 - Δw) * s[h₁, w₁] + Δw * s[h₁, w₂]) +
            Δh * ((1 - Δw) * s[h₂, w₁] + Δw * s[h₂, w₂])
        else
            s[h, w]
        end
        return
    end

    # update the `u` and `v` fields.
    kernel_u = @cuda launch = false core_u(f.f, f.u, f.v, f._tmp[1])
    kernel_v = @cuda launch = false core_v(f.f, f.u, f.v, f._tmp[3])
    threads_u = min(H * (W + 1), launch_configuration(kernel_u.fun).threads)
    threads_v = min((H + 1) * W, launch_configuration(kernel_v.fun).threads)
    CUDA.@sync begin
        kernel_u(f.f, f.u, f.v, f._tmp[1]; threads = threads_u, blocks = cld(H * (W + 1), threads_u))
        kernel_v(f.f, f.u, f.v, f._tmp[3]; threads = threads_v, blocks = cld((H + 1) * W, threads_v))
    end
    f.u .= f._tmp[1]
    f.v .= f._tmp[3]
    # update the `s` field
    kernel_s = @cuda launch = false core_s(f.f, f.u, f.v, f.s, f._tmp[5])
    threads_s = min(length(f.s), launch_configuration(kernel_s.fun).threads)
    CUDA.@sync kernel_s(f.f, f.u, f.v, f.s, f._tmp[5];
                        threads = threads_s, blocks = cld(length(f.s), threads_s))
    f.s .= f._tmp[5]
end

function update!(f::FluidGPU, Δt, n)
    "the full process of updating fluid"
    f.p .= 0
    applyincompressibility(f, Δt, n)
    applyflow(f, Δt)
end



function windtunnel(gridsize, Δc)
    # initialize the fluid simulator
    H, W = gridsize
    f = FluidGPU(gridsize, Δc)
    # build the scenes, there are a wall around and a solid object inside
    ff = trues(size(f.f))
    ff[1, :] .= ff[end, :] .= false
    ff[:, 1] .= false   # there is no wall at the right side for fluid flowing out
    halfy = 0.5 * H * Δc
    for w ∈ 1:W, h ∈ 1:H
        x² = ((w - 0.5) * Δc - 0.7 * halfy)^2
        y² = ((h - 0.5) * Δc - halfy)^2
        #if max(x², y²) ≤ 2      # solid cuboid
        if x² + y² ≤ 2          # solid cylinder
            ff[h + 1, w + 1] = false
        end
    end
    f.f .= CuArray(ff)  # apply the scenes into simulator
    # add a constant inject flow at the left wall, random number for breaking the symmetry
    f.u[:, 1] .= 1 #.+ 0.02 .* (CUDA.rand(Float32, H) .+ CUDA.rand(Float32, H) .- 1)
    # add a smoke source at the center of left wall
    smoke_wide = 2
    h₁ = round(Int, 0.5 * (H - smoke_wide / Δc))
    h₂ = round(Int, 0.5 * (H + smoke_wide / Δc))
    f.s[h₁:h₂, 1] .= 1
    # return the fluid simulator
    return f
end



###############################################################################  main


const MiB = 2^20
const MB = 10^6
const GiB = 2^30
const GB = 10^9

function nyasrkb(x)     # `x ∈ [0,1]`
    "the continue fitting funtion of `diverging_bkr_55_10_c35`"
	T = 1 / (1 + exp(27x -13.365f0))
	r₁ = -3.06321f0 + (17.6757f0 + (-37.1664f0 + (35.444f0 + -12.8f0x)x)x)x
	r₂ = 0.89331f0 + (-1.793f0 + 0.7f0x)x
	g₁ = -0.19013f0 + (0.502f0 + 0.2f0x)x
	g₂ = 0.311f0 + -0.357f0x
	b₁ = -0.23837f0 + (0.3211f0 + 0.91f0x)x
	b₂ = 0.25535f0 + -0.288f0x
	return (1 - T) .* (r₁, g₁, b₁) .+ T .* (r₂, g₂, b₂)
end


mutable struct FluidAnimation{Sub}
    # Sub = SubArray{NTuple{3, UInt8}, 3, CuArray{NTuple{3, UInt8}, 3}, Tuple{StepRange{Int64, Int64}, UnitRange{Int64}, Slice{OneTo{Int64}}}, false}
    frame_dir::String
    L::Int
    frame_idx::Int
    frames::Array{NTuple{3, UInt8}, 3}
    frames_gpu::CuArray{NTuple{3, UInt8}, 3}
    u_frames::Sub
    v_frames::Sub
    p_frames::Sub
    s_frames::Sub
    idx::Int
    stop::Bool

    function FluidAnimation(f::FluidGPU, L; buff_size = 1GB, frame_dir = mktempdir())
        isdir(frame_dir) || mkpath(frame_dir)
        @info "temporary frames dir: \"$frame_dir\""
        H, W = f.size
        fH, fW = 2 .* f.size .+ 4
        L_gpu = min(L, floor(Int, buff_size / (3 * fH * fW)))
        bg = floor.(UInt8, clamp.(nyasrkb(0.5f0) .* 256, 0f0, 255f0))
        frames = fill(bg, fH, fW, L_gpu)
        frames_gpu = CuArray(frames)
        u_frames = @view frames_gpu[H:-1:1, 1:W + 1, :]                 # `u` field on top left
        v_frames = @view frames_gpu[H + 1:-1:1, W + 3:2W + 2, :]        # `v` field on top right
        p_frames = @view frames_gpu[2H + 4:-1:H + 3, 1:W + 2, :]        # `p` field on bottom left
        s_frames = @view frames_gpu[2H + 4:-1:H + 3, W + 3:2W + 4, :]   # `s` field on bottom right
        return new{typeof(u_frames)}(frame_dir, L, 1, frames, frames_gpu, u_frames,
                                     v_frames, p_frames, s_frames, 1, false)
    end
end

function addframe!(anim::FluidAnimation, f::FluidGPU, u_ex, v_ex, p_ex, s_ex)
    anim.stop && return
    renderframe!(anim, f, u_ex, v_ex, p_ex, s_ex)
    anim.idx += 1
    # if the frames buffer is full, then save frames to disk
    rest = anim.L - anim.frame_idx + 1
    is_end = anim.idx > rest        # is the end of animation?
    if is_end || anim.idx > size(anim.frames_gpu, 3)
        n = is_end ? rest : size(anim.frames_gpu, 3)
        saveframes(anim, n)
        anim.frame_idx += n
        anim.idx = 1
    end
    is_end && stop!(anim)
end

function renderframe!(anim::FluidAnimation, f::FluidGPU, u_ex, v_ex, p_ex, s_ex)
    map_core = (x, ex) -> begin
        c01 = clamp(x / 2ex + 0.5f0, 0, 1)
        # ? cuda cannot complie `clamp.(...)` but `min.(max.(...))` can
        return floor.(UInt8, min.(max.(nyasrkb(c01) .* 256, 0f0), 255f0))
    end
    anim.u_frames[:, :, anim.idx] .= map_core.(f.u, u_ex)
    anim.v_frames[:, :, anim.idx] .= map_core.(f.v, v_ex)
    anim.p_frames[:, :, anim.idx] .= map_core.(f.p, p_ex)
    anim.s_frames[:, :, anim.idx] .= map_core.(f.s, s_ex)
end

function saveframes(anim::FluidAnimation, n)
    "save `n` frames to disk, used in `addframe!`"
    copy!(anim.frames, anim.frames_gpu)             # copy frames from gpu to cpu
    frames = reinterpret(RGB{N0f8}, anim.frames)    # reinterpret the correct data type to save image
    Threads.@threads for idx ∈ 1:n
        png_path = anim.frame_dir * (@sprintf "/%06d.png" anim.frame_idx + idx - 1) # get the frame path
        save(png_path, frames[:, :, idx])
    end
end

function stop!(anim::FluidAnimation)
    "stop adding frames and free the buffer, used in `addframe!`"
    anim.stop = true
    anim.frames = Array{NTuple{3, UInt8}}(undef, 0, 0, 0)
    anim.frames_gpu = CuArray{NTuple{3, UInt8}}(undef, 1, 1, 1)
    anim.u_frames = anim.v_frames = anim.p_frames = anim.s_frames = @view anim.frames_gpu[end:-1:1, 1:end, :]
    anim.frame_idx = anim.idx = 0
end

function savevideo(path, anim::FluidAnimation; fps = 25, crf = 28, preset = "slow")
    "save video to `path` using ffmpeg with NVENC encoding, run `FFMPEG.exe(`-h encoder=hevc_nvenc`)` to see more"
    anim.stop || @warn "saving video while the animation is not stop"
    FFMPEG.exe(`-y -v 16 -framerate $fps -i $(anim.frame_dir)/%06d.png -c:v hevc_nvenc -tune hq -rc:v vbr -cq:v $crf -preset $preset -pix_fmt yuv420p $path`)
    @info "saved video at \"$path\""
end



function main()
    frame_size = (1080, 1920)       # 1080p
    frame_time = 1/30               # simulation time between two frames
    simulate_time = 360             # total simulation time
    n_update = 4                    # n simulation updates per frame
    fps = 60
    # prepare the simulator and animation struct
    time_axis = 0:frame_time:simulate_time
    fluid = windtunnel((frame_size .- 4) .÷ 2, 0.07)
    anim = FluidAnimation(fluid, length(time_axis) + 1; buff_size = 4GB)
    # simulate and render frame
    Δt = frame_time / n_update
    ex = (1.5f0, 1f0, 0.5f0, 1f0)        # max value shown on video
    @showprogress for _ ∈ time_axis
        addframe!(anim, fluid, ex...)
        for _ ∈ 1:n_update
            update!(fluid, Δt, 100)
        end
    end
    addframe!(anim, fluid, ex...)
    # save video and remove frame dir
    savevideo("fluid.mp4", anim; fps, crf = 19)
    rm(anim.frame_dir; force = true, recursive = true)
end
main()
