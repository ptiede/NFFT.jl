"""
    Reactant_NFFTPlan

NFFT plan optimized for Reactant compilation. Uses TENSOR mode precomputation
which stores precomputed window function values for each node.
Supports D=1,2,3 dimensions.

The plan stores:
- Precomputed window tensor: (2m, D, J) array of window function values  
- Precomputed linear indices for gather/scatter operations
- Precomputed deconvolution LUT and indices

Note: NFFTParams is NOT stored to avoid Reactant tracing issues with non-traceable fields.
"""
mutable struct Reactant_NFFTPlan{T<:Number, D, M, K, WI, WP, DI, WH} <: AbstractNFFTPlan{T,D,1}
    N::NTuple{D,Int64}
    NOut::NTuple{1,Int64}
    J::Int64
    k::K
    Ñ::NTuple{D,Int64}
    dims::UnitRange{Int64}
    # Precomputed linear indices: (2m^D, J) - linear indices into flattened grid for each node
    # This allows using a single gather/scatter per node
    linearIndices::WI
    # Precomputed window product: (2m^D, J) - product of separable windows for each node
    windowProduct::WP
    # Deconvolution indices: linear indices mapping input to oversampled grid
    deconvolveIdx::DI
    # Flat deconvolution LUT: product of separable window hat inverse
    windowHatInvLUT::WH
end

# Access M (window width 2m) as compile-time constant
@inline window_width(::Reactant_NFFTPlan{T,D,M}) where {T,D,M} = M

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
    
    m = params.m
    M = 2m  # window width
    
    # Precompute window tensor, linear indices, and window products
    linearIndices, windowProduct = precompute_window_tensor_reactant(k, Ñ, params)
    
    # Precompute deconvolution LUT and indices (returns RArrays)
    deconvolveIdx, windowHatInvLUT = precompute_deconvolve_reactant(N, Ñ, params)
    
    # Convert k to RArray as well
    k_r = Reactant.to_rarray(collect(k))
    
    return Reactant_NFFTPlan{T, D, M, typeof(k_r), typeof(linearIndices), typeof(windowProduct), typeof(deconvolveIdx), typeof(windowHatInvLUT)}(
        N, NOut, J, k_r, Ñ, dims_,
         linearIndices, windowProduct, deconvolveIdx, windowHatInvLUT
    )
end

AbstractNFFTs.size_in(p::Reactant_NFFTPlan) = p.N
AbstractNFFTs.size_out(p::Reactant_NFFTPlan) = p.NOut
AbstractNFFTs.size_out(p::AdjointRPlan) = AbstractNFFTs.size_in(p.plan)
AbstractNFFTs.size_in(p::AdjointRPlan) = AbstractNFFTs.size_out(p.plan)

function Base.show(io::IO, p::Reactant_NFFTPlan{T,D,M}) where {T,D,M}
    print(io, "Reactant_NFFTPlan with $(p.J) sampling points for $(D)D input of size $(p.N), window=$(M)")
end

#############################
# Reactant tracing support
# The plan is passed as Const - arrays inside are already RArrays
#############################

# Tell Reactant to treat the plan as a constant structure with traced array fields
Base.@nospecializeinfer function Reactant.traced_type_inner(
    @nospecialize(RT::Type{<:Reactant_NFFTPlan{T,D,M,K,WI,WP,DI,WH}}),
    seen,
    mode::TraceMode,
    @nospecialize(track_numbers::Type),
    @nospecialize(ndevices),
    @nospecialize(runtime)
) where {T,D,M,K,WI,WP,DI,WH}
    K2 = traced_type_inner(K, seen, mode, track_numbers, ndevices, runtime)
    WI2 = traced_type_inner(WI, seen, mode, track_numbers, ndevices, runtime)
    WP2 = traced_type_inner(WP, seen, mode, track_numbers, ndevices, runtime)
    DI2 = traced_type_inner(DI, seen, mode, track_numbers, ndevices, runtime)
    WH2 = traced_type_inner(WH, seen, mode, track_numbers, ndevices, runtime)
    return Reactant_NFFTPlan{T, D, M, K2, WI2, WP2, DI2, WH2}
end

Base.@nospecializeinfer function Reactant.make_tracer(
    seen,
    prev::Reactant_NFFTPlan{T,D,M},
    @nospecialize(path),
    mode;
    kwargs...
) where {T,D,M}
    if mode == Reactant.TracedToTypes
        push!(path, Core.Typeof(prev))
        return nothing
    end
    
    if haskey(seen, prev)
        return seen[prev]
    end
    
    k_traced = Reactant.make_tracer(seen, prev.k, (path..., :k), mode; kwargs...)
    li_traced = Reactant.make_tracer(seen, prev.linearIndices, (path..., :linearIndices), mode; kwargs...)
    wp_traced = Reactant.make_tracer(seen, prev.windowProduct, (path..., :windowProduct), mode; kwargs...)
    di_traced = Reactant.make_tracer(seen, prev.deconvolveIdx, (path..., :deconvolveIdx), mode; kwargs...)
    wh_traced = Reactant.make_tracer(seen, prev.windowHatInvLUT, (path..., :windowHatInvLUT), mode; kwargs...)
    
    result = Reactant_NFFTPlan{T, D, M, typeof(k_traced), typeof(li_traced), typeof(wp_traced), typeof(di_traced), typeof(wh_traced)}(
        prev.N, prev.NOut, prev.J, k_traced, prev.Ñ, prev.dims,
        li_traced, wp_traced, di_traced, wh_traced
    )
    seen[prev] = result
    return result
end

#############################
# Precomputation functions
#############################

"""
Precompute window tensor, linear indices, and window products.
Returns:
- linearIndices: (2m^D, J) array of linear indices into flattened Ñ grid
- windowProduct: (2m^D, J) array of window products (outer product of separable windows)
"""
function precompute_window_tensor_reactant(k::AbstractMatrix{T}, Ñ::NTuple{D,Int}, params) where {T,D}
    m = params.m
    σ = params.σ
    J = size(k, 2)
    M = 2m
    
    win, _ = getWindow(params.window)
    P = precomputePolyInterp(win, m, σ, T)
        
    # For each node, compute M^D linear indices and window products
    numStencil = M^D
    linearIndices = zeros(Int64, numStencil, J)
    windowProduct = zeros(T, numStencil, J)
    
    # Shift nodes to [0, 1)
    kShifted = collect(k)
    shiftNodes!(kShifted)
    
    # Precompute strides for linear indexing
    strides = ntuple(d -> d == 1 ? 1 : prod(Ñ[1:d-1]), D)
    
    for j in 1:J
        # Compute separable window values and base indices for each dimension        
        # Compute all M^D combinations of indices and window products
        for (idx, ci) in enumerate(CartesianIndices(ntuple(_ -> 1:M, D)))
            # Compute linear index with wrapping
            linIdx = 1
            winProd = one(T)
            for d in 1:D
                l = ci[d]

                xtmp = kShifted[d, j]
                kscale = xtmp * Ñ[d]
                off = unsafe_trunc(Int, kscale) - m + 1
                k_ = kscale - off - m + 1 - T(0.5)

                wrapped = mod(off + l - 1, Ñ[d])  # 0-based wrapped index
                linIdx += wrapped * strides[d]
                winProd *= evalpoly(k_, ntuple(g -> P[g, l], size(P, 1)))
            end
            linearIndices[idx, j] = linIdx
            windowProduct[idx, j] = winProd
        end
    end
    
    # Convert to RArrays
    linearIndices_r = Reactant.to_rarray(linearIndices)
    windowProduct_r = Reactant.to_rarray(windowProduct)
    
    return linearIndices_r, windowProduct_r
end

"""
Precompute deconvolution lookup table and indices.
"""
function precompute_deconvolve_reactant(N::NTuple{D,Int}, Ñ::NTuple{D,Int}, params) where {D}
    T = eltype(params.σ)
    m = params.m
    σ = params.σ
    
    _, win_hat = getWindow(params.window)
    
    windowHatInvLUT_sep = Vector{Vector{T}}(undef, D)
    precomputeWindowHatInvLUT(windowHatInvLUT_sep, win_hat, N, Ñ, m, σ, T)
    
    windowHatInvLUT, deconvolveIdx = precompWindowHatInvLUT(params, N, Ñ, windowHatInvLUT_sep)
    
    windowHatInvLUT_real = real.(windowHatInvLUT)
    
    deconvolveIdx_r = Reactant.to_rarray(deconvolveIdx)
    windowHatInvLUT_r = Reactant.to_rarray(windowHatInvLUT_real)
    
    return deconvolveIdx_r, windowHatInvLUT_r
end

#############################
# Convolution (forward: g -> fHat)
# Uses batched gather - single operation instead of loop
#############################

function AbstractNFFTs.convolve!(
    p::Reactant_NFFTPlan{T,D},
    g::AbstractArray{<:Number, D},
    fHat::AbstractVector{<:Number},
) where {T,D}
    # Batched gather: g[linearIndices] gives (M^D, J) array
    # Then multiply by windowProduct and sum over first dimension
    gathered = g[p.linearIndices]  # (M^D, J)
    weighted = gathered .* p.windowProduct  # (M^D, J)
    copyto!(fHat, vec(sum(weighted, dims=1)))  # (J,)
    return fHat
end

#############################
# Convolution transpose (adjoint: fHat -> g)
# Uses single scatter-add via Reactant.Ops.scatter
#############################

function AbstractNFFTs.convolve_transpose!(
    p::Reactant_NFFTPlan{T,D},
    fHat::AbstractVector{<:Number},
    g::AbstractArray{<:Number, D},
) where {T,D}
    g .= zero(eltype(g))
    
    # Compute all contributions: windowProduct .* fHat' gives (M^D, J)
    # where fHat is broadcast across rows
    contributions = p.windowProduct .* transpose(fHat)  # (M^D, J)
    
    # Flatten to 1D for scatter
    flat_indices = vec(p.linearIndices)  # (M^D * J,)
    flat_contributions = vec(contributions)  # (M^D * J,)
    
    # Reshape to (1, M^D * J) for scatter indices format (index_vector_dim=1 means each row is an index vector)
    scatter_indices = reshape(flat_indices, 1, length(flat_indices))
    
    # Get flattened g as TracedRArray
    g_flat = Reactant.promote_to(Reactant.TracedRArray, vec(g))
    updates = Reactant.promote_to(Reactant.TracedRArray, flat_contributions)
    idx = Reactant.promote_to(Reactant.TracedRArray{Int64, 2}, scatter_indices)
    
    #TODO it would be great if Reactant could raise this
    # Use single scatter-add: scatter all contributions at once
    result = Reactant.Ops.scatter(
        +,  # add operation
        [g_flat],
        idx,
        [updates];
        update_window_dims=Int64[],
        inserted_window_dims=Int64[1],
        input_batching_dims=Int64[],
        scatter_indices_batching_dims=Int64[],
        scatter_dims_to_operand_dims=Int64[1],
        index_vector_dim=1,
    )[1]
    
    # Reshape back and copy to g
    copyto!(g, reshape(result, size(g)))
    
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
    # Use direct linear indexing
    g[p.deconvolveIdx] = f[1:length(p.windowHatInvLUT)] .* p.windowHatInvLUT
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
    # Use direct linear indexing
    f[1:length(p.windowHatInvLUT)] = g[p.deconvolveIdx] .* p.windowHatInvLUT
    return nothing
end

#############################
# mul! for forward and adjoint transforms
#############################

function LinearAlgebra.mul!(
    fHat::Reactant.AnyTracedRVector,
    p::Reactant_NFFTPlan{T,D},
    f::Reactant.AnyTracedRArray;
    verbose=false,
    timing::Union{Nothing,AbstractNFFTs.TimingStats}=nothing
) where {T,D}
    NFFT.consistencyCheck(p, f, fHat)
    
    g = similar(f, complex(eltype(f)), p.Ñ)
    fill!(g, zero(eltype(g)))
    
    t1 = @elapsed deconvolve!(p, f, g)
    t2 = @elapsed fft!(g)
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

function LinearAlgebra.mul!(
    f::Reactant.AnyTracedRArray,
    pl::AdjointRPlan,
    fHat::Reactant.AnyTracedRVector;
    verbose=false,
    timing::Union{Nothing,AbstractNFFTs.TimingStats}=nothing
)
    p = pl.plan
    NFFT.consistencyCheck(p, f, fHat)
    
    g = similar(f, complex(eltype(f)), p.Ñ)
    fill!(g, zero(eltype(g)))
    
    t1 = @elapsed convolve_transpose!(p, fHat, g)
    t2 = @elapsed bfft!(g)
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
# * operator for Reactant
#############################

function Base.:*(p::Reactant_NFFTPlan{T,D}, f::Reactant.AnyTracedRArray; kargs...) where {T,D}
    fHat = similar(f, complex(eltype(f)), size_out(p))
    mul!(fHat, p, f; kargs...)
    return fHat
end

function Base.:*(pl::AdjointRPlan, fHat::Reactant.AnyTracedRVector; kargs...)
    f = similar(fHat, complex(eltype(fHat)), size_out(pl))
    mul!(f, pl, fHat; kargs...)
    return f
end
