module NFFTReactantExt

using NFFT, NFFT.AbstractNFFTs
using NFFT: NFFTParams, indexOffset, precomputeLinInterp, precomputePolyInterp, 
            initParams, getWindow, shiftNodes!, checkNodes, TENSOR,
            precomputeWindowHatInvLUT, precompWindowHatInvLUT
using NFFT.LinearAlgebra
using NFFT.SparseArrays: sparse
using AbstractFFTs
using Reactant
using Reactant: TracedRArray, TracedRNumber, TraceMode, traced_type_inner
using ReactantCore: @trace

include("implementation.jl")

end