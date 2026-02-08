"""
    Reactant_NFFTPlan

NFFT plan optimized for Reactant compilation. Uses TENSOR mode precomputation
which stores precomputed window function values for each node.
Supports D=1,2,3 dimensions.

The plan stores:
- Precomputed window tensor: (2m, D, J) array of window function values
- Precomputed indices: (2m, D, J) array of wrapped indices for each node
- Precomputed deconvolution LUT and indices

Note: NFFTParams is NOT stored to avoid Reactant tracing issues with non-traceable fields.
"""
mutable struct Reactant_NFFTPlan{T<:Number, D, K<:AbstractMatrix, WT<:AbstractArray{<:Any,3}, WI<:AbstractArray{<:Any,3}, DI<:AbstractVector, WH<:AbstractVector} <: AbstractNFFTPlan{T,D,1}
    N::NTuple{D,Int64}
    NOut::NTuple{1,Int64}
    J::Int64
    k::K
    Ñ::NTuple{D,Int64}
    dims::UnitRange{Int64}
    # Precomputed window tensor: (2m, D, J) - window values for each node
    windowTensor::WT
    # Precomputed indices: (2m, D, J) - wrapped indices for each dimension and node
    windowIndices::WI
    # Deconvolution indices: linear indices mapping input to oversampled grid
    deconvolveIdx::DI
    # Flat deconvolution LUT: product of separable window hat inverse
    windowHatInvLUT::WH
end

# TODO figure out Adjoint Ancestor indices issue
struct AdjointRPlan{P}
    plan::P
end
Base.adjoint(p::Reactant_NFFTPlan) = AdjointRPlan(p)


"""
Create NFFT plan for Reactant arrays.
"""
function AbstractNFFTs.plan_nfft(::NFFT.NFFTBackend, ::Type{<:Reactant.RArray}, k::AbstractMatrix{T}, N::NTuple{D,Int}, rest...;
                                 timing::Union{Nothing,AbstractNFFTs.TimingStats}=nothing, kargs...) where {T,D}
    t = @elapsed begin
        p = Reactant_NFFTPlan(k, N, rest...; kargs...)
    end
    if timing !== nothing
        timing.pre = t
    end
    return p
end

function Reactant_NFFTPlan(k::AbstractMatrix{T}, N::NTuple{D,Int}; dims::Union{Integer,UnitRange{Int64}}=1:D,
                           fftflags=nothing, kwargs...) where {T,D}
    D > 3 && throw(ArgumentError("Reactant NFFT only supports D ≤ 3, got D=$D"))
    
    checkNodes(k)
    
    # Force TENSOR precompute for Reactant
    params, N, NOut, J, Ñ, dims_ = initParams(k, N, dims; precompute=TENSOR, 
                                              storeDeconvolutionIdx=true, 
                                              blocking=false, kwargs...)
    
    if length(dims_) != D
        error("Reactant NFFT does not support directional transforms yet!")
    end
    
    # Precompute window tensor and indices (returns RArrays)
    windowTensor, windowIndices = precompute_window_tensor_reactant(k, Ñ, params)
    
    # Precompute deconvolution LUT and indices (returns RArrays)
    deconvolveIdx, windowHatInvLUT = precompute_deconvolve_reactant(N, Ñ, params)
    
    # Convert k to RArray as well
    k_r = Reactant.to_rarray(collect(k))
    
    return Reactant_NFFTPlan{T, D, typeof(k_r), typeof(windowTensor), typeof(windowIndices), typeof(deconvolveIdx), typeof(windowHatInvLUT)}(
        N, NOut, J, k_r, Ñ, dims_,
        windowTensor, windowIndices, deconvolveIdx, windowHatInvLUT
    )
end

AbstractNFFTs.size_in(p::Reactant_NFFTPlan) = p.N
AbstractNFFTs.size_out(p::Reactant_NFFTPlan) = p.NOut
AbstractNFFTs.size_out(p::AdjointRPlan) = AbstractNFFTs.size_in(p.plan)
AbstractNFFTs.size_in(p::AdjointRPlan) = AbstractNFFTs.size_out(p.plan)

function Base.show(io::IO, p::Reactant_NFFTPlan{T,D}) where {T,D}
    print(io, "Reactant_NFFTPlan with $(p.J) sampling points for $(D)D input of size $(p.N)")
end

#############################
# Reactant tracing support
# Only array fields should be traced; N, NOut, J, Ñ, dims remain constants
#############################

# Tell Reactant the traced type should keep T, D unchanged and trace the array type parameters
Base.@nospecializeinfer function Reactant.traced_type_inner(
    @nospecialize(RT::Type{<:Reactant_NFFTPlan{T,D}}),
    seen,
    mode::TraceMode,
    @nospecialize(track_numbers::Type),
    @nospecialize(ndevices),
    @nospecialize(runtime)
) where {T,D}
    # T and D are constant type parameters
    # Only trace the array type parameters (K, WT, WI, DI, WH)
    K2 = traced_type_inner(RT.parameters[3], seen, mode, track_numbers, ndevices, runtime)
    WT2 = traced_type_inner(RT.parameters[4], seen, mode, track_numbers, ndevices, runtime)
    WI2 = traced_type_inner(RT.parameters[5], seen, mode, track_numbers, ndevices, runtime)
    DI2 = traced_type_inner(RT.parameters[6], seen, mode, track_numbers, ndevices, runtime)
    WH2 = traced_type_inner(RT.parameters[7], seen, mode, track_numbers, ndevices, runtime)
    return Reactant_NFFTPlan{T, D, K2, WT2, WI2, DI2, WH2}
end

# Custom make_tracer to only trace array fields, keeping scalars/tuples constant
Base.@nospecializeinfer function Reactant.make_tracer(
    seen,
    prev::Reactant_NFFTPlan{T,D},
    @nospecialize(path),
    mode;
    kwargs...
) where {T,D}
    if mode == Reactant.TracedToTypes
        # Just register that we visited this type
        push!(path, Core.Typeof(prev))
        return nothing
    end
    
    if haskey(seen, prev)
        return seen[prev]
    end
    
    # Only trace the array fields, keep scalar/tuple fields as constants
    k_traced = Reactant.make_tracer(seen, prev.k, (path..., :k), mode; kwargs...)
    wt_traced = Reactant.make_tracer(seen, prev.windowTensor, (path..., :windowTensor), mode; kwargs...)
    wi_traced = Reactant.make_tracer(seen, prev.windowIndices, (path..., :windowIndices), mode; kwargs...)
    di_traced = Reactant.make_tracer(seen, prev.deconvolveIdx, (path..., :deconvolveIdx), mode; kwargs...)
    wh_traced = Reactant.make_tracer(seen, prev.windowHatInvLUT, (path..., :windowHatInvLUT), mode; kwargs...)
    
    result = Reactant_NFFTPlan{T, D, typeof(k_traced), typeof(wt_traced), typeof(wi_traced), typeof(di_traced), typeof(wh_traced)}(
        prev.N,      # constant - not traced
        prev.NOut,   # constant - not traced
        prev.J,      # constant - not traced
        k_traced,
        prev.Ñ,      # constant - not traced
        prev.dims,   # constant - not traced
        wt_traced,
        wi_traced,
        di_traced,
        wh_traced
    )
    seen[prev] = result
    return result
end

#############################
# Precomputation functions
#############################

"""
Precompute window tensor and wrapped indices for TENSOR mode.
Uses NFFT's existing polynomial interpolation, then converts to RArrays.
Returns:
- windowTensor: (2m, D, J) array of window function values
- windowIndices: (2m, D, J) array of wrapped indices
"""
function precompute_window_tensor_reactant(k::AbstractMatrix{T}, Ñ::NTuple{D,Int}, params) where {T,D}
    m = params.m
    σ = params.σ
    J = size(k, 2)
    m2 = 2m
    
    win, _ = getWindow(params.window)
    
    # Use NFFT's existing polynomial interpolation
    P = precomputePolyInterp(win, m, σ, T)
    
    windowTensor = zeros(T, m2, D, J)
    windowIndices = zeros(Int64, m2, D, J)
    
    # Shift nodes to [0, 1) - work on a copy
    kShifted = collect(k)  # ensure it's a regular Array for mutation
    shiftNodes!(kShifted)
    itrD = 1:D
    itrm = 1:m2
    itrJ = 1:J
    @trace for j in itrJ
        @trace for d in itrD
            xtmp = kShifted[d, j]
            kscale = xtmp * Ñ[d]
            off = unsafe_trunc(Int, kscale) - m + 1
            
            # Compute polynomial evaluation point (same as NFFT._precomputeWindowTensor)
            k_ = kscale - off - m + 1 - T(0.5)
            
            @trace for l in itrm
                # Store wrapped index (1-based)
                windowIndices[l, d, j] = mod(off + l - 1, Ñ[d]) + 1
                
                # Compute window value using polynomial interpolation
                windowTensor[l, d, j] = evalpoly(k_, ntuple(g -> P[g, l], size(P, 1)))
            end
        end
    end
    
    # Convert to RArrays for Reactant tracing
    windowTensor_r = Reactant.to_rarray(windowTensor)
    windowIndices_r = Reactant.to_rarray(windowIndices)
    
    return windowTensor_r, windowIndices_r
end

"""
Precompute deconvolution lookup table and indices using NFFT's existing functions.
"""
function precompute_deconvolve_reactant(N::NTuple{D,Int}, Ñ::NTuple{D,Int}, params) where {D}
    T = eltype(params.σ)
    m = params.m
    σ = params.σ
    
    _, win_hat = getWindow(params.window)
    
    # Use NFFT's existing function to compute separable LUTs
    windowHatInvLUT_sep = Vector{Vector{T}}(undef, D)
    precomputeWindowHatInvLUT(windowHatInvLUT_sep, win_hat, N, Ñ, m, σ, T)
    
    # Use NFFT's existing function to compute flat LUT and indices
    windowHatInvLUT, deconvolveIdx = precompWindowHatInvLUT(params, N, Ñ, windowHatInvLUT_sep)
    
    # Convert to real (NFFT returns Complex but values are real for deconvolution)
    windowHatInvLUT_real = real.(windowHatInvLUT)
    
    # Convert to RArrays for Reactant tracing
    deconvolveIdx_r = Reactant.to_rarray(deconvolveIdx)
    windowHatInvLUT_r = Reactant.to_rarray(windowHatInvLUT_real)
    
    return deconvolveIdx_r, windowHatInvLUT_r
end

#############################
# Convolution (forward: g -> fHat)
#############################

function AbstractNFFTs.convolve!(
    p::Reactant_NFFTPlan{T,D},
    g::AbstractArray{<:Number, D},
    fHat::AbstractVector{<:Number},
) where {T,D}
    convolve_tensor_reactant!(p, g, fHat)
    return fHat
end

# 1D convolution - vectorized sum over window
function convolve_tensor_reactant!(p::Reactant_NFFTPlan{T,1}, g, fHat) where {T}
    @allowscalar @trace for j in 1:p.J
        idx = p.windowIndices[:, 1, j]
        win = p.windowTensor[:, 1, j]
        fHat[j] = sum(win .* g[idx])
    end
    return fHat
end

# 2D convolution - vectorized over window
function convolve_tensor_reactant!(p::Reactant_NFFTPlan{T,2}, g, fHat) where {T}
    @allowscalar@trace for j in 1:p.J
        idx1 = p.windowIndices[:, 1, j]
        idx2 = p.windowIndices[:, 2, j]
        win1 = p.windowTensor[:, 1, j]
        win2 = p.windowTensor[:, 2, j]
        # Outer product of windows times gathered subgrid
        fHat[j] = sum(win1 .* (transpose(win2) .* g[idx1, idx2]))
    end
    return fHat
end

# 3D convolution - vectorized over window
function convolve_tensor_reactant!(p::Reactant_NFFTPlan{T,3}, g, fHat) where {T}
    @allowscalar@trace for j in 1:p.J
        idx1 = p.windowIndices[:, 1, j]
        idx2 = p.windowIndices[:, 2, j]
        idx3 = p.windowIndices[:, 3, j]
        win1 = p.windowTensor[:, 1, j]
        win2 = p.windowTensor[:, 2, j]
        win3 = p.windowTensor[:, 3, j]
        # 3D outer product of windows
        win_3d = reshape(win1, :, 1, 1) .* reshape(win2, 1, :, 1) .* reshape(win3, 1, 1, :)
        fHat[j] = sum(win_3d .* g[idx1, idx2, idx3])
    end
    return fHat
end

#############################
# Convolution transpose (adjoint: fHat -> g)
#############################

function AbstractNFFTs.convolve_transpose!(
    p::Reactant_NFFTPlan{T,D},
    fHat::AbstractVector{<:Number},
    g::AbstractArray{<:Number, D},
) where {T,D}
    g .= zero(eltype(g))
    convolve_transpose_tensor_reactant!(p, fHat, g)
    return g
end

# 1D convolution transpose - scatter with broadcasting
function convolve_transpose_tensor_reactant!(p::Reactant_NFFTPlan{T,1}, fHat, g) where {T}
    @allowscalar @trace for j in 1:p.J
        idx = p.windowIndices[:, 1, j]
        win = p.windowTensor[:, 1, j]
        g[idx] = g[idx] .+ win .* fHat[j]
    end
    return g
end

# 2D convolution transpose - scatter with broadcasting
function convolve_transpose_tensor_reactant!(p::Reactant_NFFTPlan{T,2}, fHat, g) where {T}
    @allowscalar @trace for j in 1:p.J
        idx1 = p.windowIndices[:, 1, j]
        idx2 = p.windowIndices[:, 2, j]
        win1 = p.windowTensor[:, 1, j]
        win2 = p.windowTensor[:, 2, j]
        # Outer product of windows times scalar
        g[idx1, idx2] = g[idx1, idx2] .+ (win1 .* transpose(win2)) .* fHat[j]
    end
    return g
end

# 3D convolution transpose - scatter with broadcasting
function convolve_transpose_tensor_reactant!(p::Reactant_NFFTPlan{T,3}, fHat, g) where {T}
    @allowscalar @trace for j in 1:p.J
        idx1 = p.windowIndices[:, 1, j]
        idx2 = p.windowIndices[:, 2, j]
        idx3 = p.windowIndices[:, 3, j]
        win1 = p.windowTensor[:, 1, j]
        win2 = p.windowTensor[:, 2, j]
        win3 = p.windowTensor[:, 3, j]
        # 3D outer product of windows
        win_3d = reshape(win1, :, 1, 1) .* reshape(win2, 1, :, 1) .* reshape(win3, 1, 1, :)
        g[idx1, idx2, idx3] = g[idx1, idx2, idx3] .+ win_3d .* fHat[j]
    end
    return g
end

#############################
# Deconvolution (f -> g)
#############################

function AbstractNFFTs.deconvolve!(
    p::Reactant_NFFTPlan{T,D},
    f::AbstractArray{<:Number, D},
    g::AbstractArray{<:Number, D}
) where {T,D}
    # Broadcasting handles buffer donation automatically
    g[p.deconvolveIdx] = vec(f) .* p.windowHatInvLUT
    return nothing
end

#############################
# Deconvolution transpose (g -> f)
#############################

function AbstractNFFTs.deconvolve_transpose!(
    p::Reactant_NFFTPlan{T,D},
    g::AbstractArray{<:Number, D},
    f::AbstractArray{<:Number, D}
) where {T,D}
    # Broadcasting handles buffer donation automatically
    copyto!(f, reshape(g[p.deconvolveIdx] .* p.windowHatInvLUT, size(f)))
    return nothing
end

#############################
# mul! for forward and adjoint transforms
#############################

"""
Forward NFFT: f -> fHat
"""
function LinearAlgebra.mul!(
    fHat::Reactant.AnyTracedRVector,
    p::Reactant_NFFTPlan{T,D},
    f::Reactant.AnyTracedRArray;
    verbose=false,
    timing::Union{Nothing,AbstractNFFTs.TimingStats}=nothing
) where {T,D}
    NFFT.consistencyCheck(p, f, fHat)
    
    # Create temp array for oversampled grid (zeros via broadcasting)
    g = similar(f, complex(eltype(f)), p.Ñ)
    g .= zero(eltype(g))
    
    t1 = @elapsed deconvolve!(p, f, g)
    t2 = @elapsed begin
        # In-place FFT using AbstractFFTs (works with Reactant)
        fft!(g)
    end
    t3 = @elapsed convolve!(p, g, fHat)
    
    if verbose
        @info "Timing: deconv=$t1 fft=$t2 conv=$t3"
    end
    if timing !== nothing
        timing.conv = t3
        timing.fft = t2
        timing.deconv = t1
    end
    
    return fHat
end

"""
Adjoint NFFT: fHat -> f
"""
function LinearAlgebra.mul!(
    f::Reactant.AnyTracedRArray,
    pl::AdjointRPlan,
    fHat::Reactant.AnyTracedRVector;
    verbose=false,
    timing::Union{Nothing,AbstractNFFTs.TimingStats}=nothing
)
    p = pl.plan
    NFFT.consistencyCheck(p, f, fHat)
    
    # Create temp array for oversampled grid (zeros via broadcasting)
    g = similar(f, complex(eltype(f)), p.Ñ)
    g .= zero(eltype(g))
    
    t1 = @elapsed convolve_transpose!(p, fHat, g)
    t2 = @elapsed begin
        # Backward FFT using AbstractFFTs (works with Reactant)
        bfft!(g)
    end
    t3 = @elapsed deconvolve_transpose!(p, g, f)
    
    if verbose
        @info "Timing: conv=$t1 fft=$t2 deconv=$t3"
    end
    if timing !== nothing
        timing.conv_adjoint = t1
        timing.fft_adjoint = t2
        timing.deconv_adjoint = t3
    end
    
    return f
end

#############################
# * operator for Reactant (looser type constraints than AbstractNFFTs)
#############################

"""
Forward NFFT via * operator: p * f -> fHat
"""
function Base.:*(p::Reactant_NFFTPlan{T,D}, f::Reactant.AnyTracedRArray; kargs...) where {T,D}
    fHat = similar(f, complex(eltype(f)), size_out(p))
    mul!(fHat, p, f; kargs...)
    return fHat
end

"""
Adjoint NFFT via * operator: p' * fHat -> f
"""
function Base.:*(pl::AdjointRPlan, fHat::Reactant.AnyTracedRVector; kargs...)
    f = similar(fHat, complex(eltype(fHat)), size_out(pl))
    mul!(f, pl, fHat; kargs...)
    return f
end