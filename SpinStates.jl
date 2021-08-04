# # Spin States
module SpinStates

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
struct States{T<:Complex}
    buffers::Vector{Tuple{AbstractArray{T}, StructArray{State{T}}}}
    originIndex::Int
    function States{T}(m::Int, n::Int) where
        {T<:Complex, AT<:AbstractArray{T, 3}}
        @assert isodd(m) "m dimension of States must be odd"
        tempA = zeros(T, m, n, 3)
        tempSA = StructArray{State{T}}(tempA, dims=3)
        tempB = zeros(T, m, n, 3)
        tempSB = StructArray{State{T}}(tempB, dims=3)
        new([(tempA, tempSA), (tempB, tempSB)], 1 + ((m - 1) / 2))
        #fPlusInit = zeros(T, m, n)
        #fMinusInit = similar(fPlusInit)
        #zInit = similar(fPlusInit)
        #tempSA = StructArray{State{T}}(
        #    fPlus = fPlusInit,
        #    fMinus = fMinusInit,
        #    z = zInit)
        #new([tempSA, similar(tempSA)], 1 + ((m - 1) / 2))
    end
end

function fullyrelaxstates!(s::S) where
    {S <: States}
    fill!(s.buffers[1][1], 0.0)
    s.buffers[1][2].z[s.originIndex,:] .= 1.0
    nothing
end



# We can do that actual buffer-swap by just reversing the order of the buffers
function swapbuffers!(s::States{T}) where
        {T<:Complex, AT<:AbstractArray}
    reverse!(s.buffers)
    nothing
end
    
##Excitation operation
struct Excitation{T<:AbstractFloat}

    opMat::DenseArray{Complex{T}, 2}

    function Excitation(flipAngle::T, phase::T) where T<:AbstractFloat
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
        new{T}(tempMat)
    end
end

function (f::Excitation)(s::States)
    mul!(reshape(s.buffers[2][1],(:,3)), reshape(s.buffers[1][1], (:,3)), f.opMat)
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

struct SpinEnvironmentScales{T<:AbstractFloat}
    scaleT1::T
    scaleT2::T
    addT1::T
end

struct Relaxation{T<:AbstractFloat}
    relaxationScales::Tuple{
        AbstractArray{T}, StructVector{SpinEnvironmentScales{T}}}
    function Relaxation{T}(e::Environments{T}, duration::T) where
        T<:AbstractFloat
        tempA = zeros(T, size(e.relaxationConstants)[1], 3)
        tempSA = StructVector{SpinEnvironmentScales{T}}(tempA, dims=2)
        tempSA.scaleT1 .= exp.(-duration ./ e.relaxationConstants.t1)
        tempSA.scaleT2 .= exp.(-duration ./ e.relaxationConstants.t2)
        tempSA.addT1 .= 1.0 .- tempSA.scaleT1
        new((tempA, tempSA))
    end  
end

function (f::Relaxation)(s::States)
    broadcast!(*,
        s.buffers[2][2].fPlus,
        reshape(f.relaxationScales[2].scaleT2, 1, :),
        s.buffers[1][2].fPlus)
    broadcast!(*,
        s.buffers[2][2].fMinus, 
        reshape(f.relaxationScales[2].scaleT2, 1, :),
        s.buffers[1][2].fMinus)
    broadcast!(*,
        s.buffers[2][2].z,
        reshape(f.relaxationScales[2].scaleT1, 1, :),
        s.buffers[1][2].z) 
    s.buffers[2][2].z[s.originIndex, :] .+= f.relaxationScales[2].addT1
    swapbuffers!(s)
    nothing
end

struct Spoiling
    spoilGrad::Int

    function Spoiling(sg::Int)
        new(sg)
    end
end

function (f::Spoiling)(s::States)
    circshift!(
        view(s.buffers[2][1], :, :,  1),
        view(s.buffers[1][1], :, :, 1),
        (f.spoilGrad, 0))
    
    circshift!(
        view(s.buffers[2][1], :, :, 2),
        view(s.buffers[1][1], :, :, 2),
        (-f.spoilGrad, 0))
   
    if f.spoilGrad < 0
        last = size(s.buffers[1][1])[1] 
        ind = last - f.spoilGrad + 1
        view(s.buffers[2][1], ind:last, :, 1)[:] .= 0.0
        view(s.buffers[2][1], 1:f.spoilGrad, :, 2)[:] .= 0.0
    elseif f.spoilGrad >0
        last = size(s.buffers[1][1])[1] 
        ind = last - f.spoilGrad + 1
        view(s.buffers[2][1], 1:f.spoilGrad, :, 1)[:] .= 0.0
        view(s.buffers[2][1], ind:last, :, 2)[:] .= 0.0
    end
    

    s.buffers[2][1][:, :, 3] .= s.buffers[1][1][:, :, 3]
 
    swapbuffers!(s)
    nothing
end

end
