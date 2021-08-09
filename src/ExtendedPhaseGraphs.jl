# # ExtendedPhaseGraphs
module ExtendedPhaseGraphs

using LinearAlgebra

using StructArrays

## Individual spin states
# Spin states, when represeted as a single object, are simply a struct with the
# the $F^+$, $F^-$, and $Z$ components available as complex values.
struct State{T<:Complex}
    fPlus::T
    fMinus::T
    z::T
end

## Aggregate spin states
# The total of all $M$ spin states, each contained within $N$ spin conditions,
# are represented as an $M \times N \times 3$ array, organized in memory such
# that all $F^+$ entries go first, followed by all $F^-$ entries, and finally all
# $Z$ entries. Moreover, the order of $F^-$ entries is reversed, such that the
# $F^-(-k)$ state appears at the same index as $F^(k)$ and $Z(k)$. This layout
# is designed to allow excitation operations to be implemented efficiently.
#
# Also, in the name of efficiency, we keep two buffers: a front buffer that
# represents the current state of the system, and a back buffer that can be used
# as the destination for out-of-place operations mutating the state of the
# system. We can then just do a buffer-swap via exchanging pointers to these
# buffers, without requiring us to copy the whole system state.
#
# Each buffer is represented both by a flat array in an efficient memory layout
# and a StructArray that provides a convience (inefficient) view for accessing
# individual entries. For more details on this, you can see this discussion:
# https://discourse.julialang.org/t/structarrays-memory-layout/65105
mutable struct States{
    T<:Complex,
    AT <: AbstractArray{T, 3},
    FT <: AbstractArray{T, 2}
    }
    frontbuffer::AT
    frontbufferflatview::FT
    backbuffer::AT
    backbufferflatview::FT
    originIndex::Int
    function States{T, AT, FT}(m::Int, n::Int) where
        {T<:Complex,
        AT<:AbstractArray{T, 3},
        FT <: AbstractArray{T, 2}
        }
        @assert isodd(m) "m dimension of States must be odd"
        tempA = zeros(T, m, n, 3)
        
        tempB = zeros(T, m, n, 3)
        
        new(
            tempA,
            reshape(tempA,(:,3)),
            tempB,
            reshape(tempB,(:,3)),
            1 + ((m - 1) / 2))
    end
end

@inline @views function fplus(s::States)
    @inbounds s.frontbuffer[:,:,1]
end

@inline @views function fminus(s::States)
    @inbounds s.frontbuffer[:,:,2]
end

@inline @views function z(s::States)
    @inbounds s.frontbuffer[:,:,3]
end

@inline @views function fplusbackbuffer(s::States)
    @inbounds s.backbuffer[:,:,1]
end

@inline @views function fminusbackbuffer(s::States)
    @inbounds s.backbuffer[:,:,2]
end

@inline @views function zbackbuffer(s::States)
    s.backbuffer[:,:,3]
end

@views function fullyrelaxstates!(s::States)
    fill!(s.frontbuffer, 0.0)
    #fill!(s.frontbuffer[s.originIndex,:,3], 1.0)
    fill!(z(s)[s.originIndex,:], 1.0)
    nothing
end



# We can do that actual buffer-swap by just reversing the order of the buffers
function swapbuffers!(s::States{T}) where {T<:Complex}
    s.frontbuffer, s.frontbufferflatview,
        s.backbuffer, s.backbufferflatview =
        s.backbuffer, s.backbufferflatview,
        s.frontbuffer, s.frontbufferflatview
    nothing
end
    
##Excitation operation
struct Excitation{T<:AbstractFloat, AT <: AbstractArray{Complex{T}, 2}}

    opMat::AT

    function Excitation{T, AT}(flipAngle::T, phase::T) where
        {T<:AbstractFloat, AT<:AbstractArray{Complex{T}, 2}}
        cosAlpha = cos(flipAngle)
        sinAlpha = sin(flipAngle)
        alphaDiv2 = flipAngle/2.0
        cosAlphaDiv2 = cos(alphaDiv2)
        cosAlphaDiv2Sq = cosAlphaDiv2^2
        sinAlphaDiv2 = sin(alphaDiv2)
        sinAlphaDiv2Sq = sinAlphaDiv2^2
        expPhase = exp(phase * (1.0im))
        expPhaseSq = exp(phase * (2.0im))
        expPhaseInv = exp(phase * (-1.0im))
        expPhaseInvSq = exp(phase * (-2.0im))

        tempMat = zeros(Complex{T}, 3, 3)
        tempMat[1] = cosAlphaDiv2Sq
        tempMat[2] = expPhaseSq * sinAlphaDiv2Sq
        tempMat[3] = (-1.0im) * expPhase * sinAlpha
        tempMat[4] = expPhaseInvSq * sinAlphaDiv2Sq
        tempMat[5] = cosAlphaDiv2Sq
        tempMat[6] = (1.0im) * expPhaseInv * sinAlpha
        tempMat[7] = (-0.5im) * expPhaseInv * sinAlpha
        tempMat[8] = (0.5im) * expPhase * sinAlpha
        tempMat[9] = cosAlpha
        new{T,AT}(tempMat)
    end
end


function (f::Excitation)(s::States)
    mul!(s.backbufferflatview, s.frontbufferflatview, f.opMat)
    swapbuffers!(s)
    nothing
end

## Individual spin environments
# Spin environments, represented as a single object are just a struct
# with the relaxation parameters available as floating point values.
struct SpinEnvironment{T<:AbstractFloat}
    t1::T
    t2::T
end

struct Environments{T<:AbstractFloat}
    relaxationConstants::StructVector{SpinEnvironment{T}}
    function Environments{T}(e::Vector{SpinEnvironment{T}}) where
        T<:AbstractFloat
        new(StructVector(e)) 
    end  
end

struct Relaxation{
    T<:AbstractFloat,
    VT <: AbstractVector{T},
    AT <: AbstractArray{T, 2}}
    scalet1::VT
    scalet1matrixview::AT
    scalet2::VT
    scalet2matrixview::AT
    addt1::VT
    function Relaxation{T, VT, AT}(e::Environments{T}, duration::T) where
        { T<:AbstractFloat, VT <: AbstractVector{T}, AT <: AbstractArray{T, 2} }
        scalet1 = zeros(T, size(e.relaxationConstants)[1])
        scalet2 = similar(scalet1) 
        addt1 = similar(scalet1) 
        scalet1 .= exp.(-duration ./ e.relaxationConstants.t1)
        scalet2 .= exp.(-duration ./ e.relaxationConstants.t2)
        addt1 .= 1.0 .- scalet1
        new(
            scalet1,
            reshape(scalet1, 1, :),
            scalet2,
            reshape(scalet2, 1, :),
            addt1)
    end  
end

@views function (f::Relaxation)(s::States)
    broadcast!(*,
        fplusbackbuffer(s),
        f.scalet2matrixview,
        fplus(s))
    broadcast!(*,
        fminusbackbuffer(s), 
        f.scalet2matrixview,
        fminus(s))
    broadcast!(*,
        zbackbuffer(s),
        f.scalet1matrixview,
        z(s)) 
    zbackbuffer(s)[s.originIndex, :] .+= f.addt1
    swapbuffers!(s)
    nothing
end

struct Spoiling
    spoilGrad::Int

    function Spoiling(sg::Int)
        new(sg)
    end
end

@views function (f::Spoiling)(s::States)
    circshift!(
        fplusbackbuffer(s),
        fplus(s),
        (f.spoilGrad, 0))
    
    circshift!(
        fminusbackbuffer(s),
        fminus(s),
        (-f.spoilGrad, 0))

    if f.spoilGrad < 0
        last = size(s.frontbuffer)[1]
        ind = last - f.spoilGrad + 1
        fplusbackbuffer(s)[ind:last, :] .= 0.0
        fminusbackbuffer(s)[1:f.spoilGrad, :] .= 0.0
    elseif f.spoilGrad >0
        last = size(s.frontbuffer)[1] 
        ind = last - f.spoilGrad + 1
        fplusbackbuffer(s)[1:f.spoilGrad, :] .= 0.0
        fminusbackbuffer(s)[ind:last, :] .= 0.0
    end
    
    copyto!(zbackbuffer(s), z(s))
 
    swapbuffers!(s)
    nothing 
end

end
